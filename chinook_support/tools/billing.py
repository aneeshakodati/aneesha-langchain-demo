"""Billing tools — order history, refund adjudication, refund execution.

Note what is missing from every signature below: a customer identifier. There is
no `customer_id` parameter to set, so there is no way for the model to fetch
anyone else's orders. Identity comes from `runtime.context`, which the model
cannot write to. See `context.py`.

`order_id` *is* a parameter, because it's the one identifier a customer
legitimately knows and types. That makes it attacker-controlled, so every tool
that accepts one re-checks ownership via `db.owns_invoice` before returning a
single field.
"""

from __future__ import annotations

from datetime import datetime

from langchain.tools import ToolRuntime, tool

from ..context import require_customer_id
from ..db import money, owns_invoice, query, query_one, write_conn
from ..policy import adjudicate, refund_history


@tool
def list_my_orders(runtime: ToolRuntime, limit: int = 10) -> dict:
    """List the customer's own recent orders, newest first.

    Use this to answer "what have I bought", "show me my orders", or to find an
    order number before looking at its details.

    Args:
        limit: How many orders to return (1-25).
    """
    customer_id = require_customer_id(runtime.context)
    limit = max(1, min(int(limit), 25))

    rows = query(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.Total, i.BillingCity, i.BillingCountry,
               (SELECT COUNT(*) FROM InvoiceLine il WHERE il.InvoiceId = i.InvoiceId)
                   AS TrackCount,
               (SELECT COUNT(*) FROM Refund r WHERE r.InvoiceId = i.InvoiceId)
                   AS RefundCount
          FROM Invoice i
         WHERE i.CustomerId = ?
         ORDER BY i.InvoiceDate DESC
         LIMIT ?
        """,
        (customer_id, limit),
    )

    now = datetime.now()
    orders = []
    for r in rows:
        placed = datetime.strptime(r["InvoiceDate"][:19], "%Y-%m-%d %H:%M:%S")
        orders.append(
            {
                "order_id": r["InvoiceId"],
                "placed_on": r["InvoiceDate"][:10],
                "days_ago": (now - placed).days,
                "total": str(money(r["Total"])),
                "track_count": r["TrackCount"],
                "refunded": bool(r["RefundCount"]),
            }
        )
    return {"order_count": len(orders), "orders": orders}


@tool
def get_order_detail(order_id: int, runtime: ToolRuntime) -> dict:
    """Show the line items on one of the customer's own orders.

    Use this when the customer asks what was on a specific order, or disputes a
    charge and you need to see what they were billed for.

    Args:
        order_id: The order number, as shown by list_my_orders.
    """
    customer_id = require_customer_id(runtime.context)

    # Ownership check. Returns the same message for "doesn't exist" and "belongs
    # to someone else" so the response can't be used to probe for valid order ids.
    if not owns_invoice(customer_id, order_id):
        return {
            "error": "not_found",
            "message": (
                f"Order #{order_id} was not found on this account. Please check "
                "the order number."
            ),
        }

    invoice = query_one(
        "SELECT InvoiceId, InvoiceDate, Total, BillingAddress, BillingCity, "
        "BillingCountry FROM Invoice WHERE InvoiceId = ?",
        (order_id,),
    )
    lines = query(
        """
        SELECT t.Name AS Title, ar.Name AS Artist, al.Title AS Album,
               COALESCE(g.Name,'Unknown') AS Genre, il.UnitPrice, il.Quantity
          FROM InvoiceLine il
          JOIN Track t   ON t.TrackId   = il.TrackId
          JOIN Album al  ON al.AlbumId  = t.AlbumId
          JOIN Artist ar ON ar.ArtistId = al.ArtistId
          LEFT JOIN Genre g ON g.GenreId = t.GenreId
         WHERE il.InvoiceId = ?
         ORDER BY il.InvoiceLineId
        """,
        (order_id,),
    )

    return {
        "order_id": order_id,
        "placed_on": invoice["InvoiceDate"][:10],
        "total": str(money(invoice["Total"])),
        "billed_to": f"{invoice['BillingCity']}, {invoice['BillingCountry']}",
        "items": [
            {
                "title": ln["Title"],
                "artist": ln["Artist"],
                "album": ln["Album"],
                "genre": ln["Genre"],
                "price": str(money(ln["UnitPrice"])),
                "quantity": ln["Quantity"],
            }
            for ln in lines
        ],
    }


@tool
def check_refund_eligibility(order_id: int, runtime: ToolRuntime) -> dict:
    """Determine whether an order can be refunded, under store policy.

    ALWAYS call this before discussing a refund. It returns the authoritative
    decision; do not reason about eligibility yourself and do not quote different
    thresholds than the ones it returns. The possible decisions are:

      auto_approve         - refundable now, no approval needed
      needs_human_approval - refundable, but a representative must sign off
      deny                 - not refundable; check requires_escalation

    If `requires_escalation` is true, say you cannot resolve it yourself and stop
    so the case can be handed to a support representative.

    Args:
        order_id: The order the customer wants refunded.
    """
    customer_id = require_customer_id(runtime.context)
    verdict = adjudicate(order_id, customer_id)
    result = verdict.to_dict()

    # Repeat-refund context, so a representative reviewing the case sees the pattern.
    prior = refund_history(customer_id)
    result["prior_refunds_on_account"] = len(prior)
    return result


@tool
def issue_refund(order_id: int, reason: str, runtime: ToolRuntime) -> dict:
    """Refund an order. This moves real money and cannot be undone by the agent.

    Only call this after `check_refund_eligibility` returned `auto_approve` or
    `needs_human_approval`. Calls for orders above the auto-approval limit will
    pause for a human representative to approve or reject.

    Args:
        order_id: The order to refund.
        reason: Why the customer is asking, in their own words. Recorded.
    """
    customer_id = require_customer_id(runtime.context)

    # Re-adjudicate at execution time rather than trusting that the model called
    # the eligibility tool first, or that the answer hasn't changed since it did.
    # A human approving the interrupt approves *this* refund, not an earlier quote.
    verdict = adjudicate(order_id, customer_id)
    if verdict.decision == "deny":
        return {
            "refunded": False,
            "reason_code": verdict.reason_code,
            "message": verdict.reason,
            "requires_escalation": verdict.requires_escalation,
        }

    approved_by = (
        "policy:auto" if verdict.decision == "auto_approve" else "human:approved"
    )
    now = datetime.now().isoformat(sep=" ", timespec="seconds")

    with write_conn() as conn:
        cur = conn.execute(
            "INSERT INTO Refund (InvoiceId, CustomerId, Amount, Reason, "
            "ApprovedBy, CreatedAt) VALUES (?, ?, ?, ?, ?, ?)",
            (
                order_id,
                customer_id,
                float(verdict.refundable_amount),
                reason,
                approved_by,
                now,
            ),
        )
        refund_id = cur.lastrowid

    return {
        "refunded": True,
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": str(verdict.refundable_amount),
        "approved_by": approved_by,
        "message": (
            f"Refunded ${verdict.refundable_amount} for order #{order_id}. "
            "It should appear on the original payment method in 5-10 business days."
        ),
    }


BILLING_TOOLS = [
    list_my_orders,
    get_order_detail,
    check_refund_eligibility,
    issue_refund,
]
