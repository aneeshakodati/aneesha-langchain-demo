"""Refund policy engine — pure functions, no LLM, no I/O beyond reads.

Why this exists as code rather than as prompt text:

A threshold written into a system prompt ("refunds over $10 need approval") is a
suggestion the model follows most of the time. Nobody can tell you what "most" is
without measuring, and it silently changes when someone rewords a neighbouring
sentence. The same threshold written here is a rule, and `adjudicate()` returns
the same verdict every time for the same inputs.

That determinism buys two things:
  1. The human-approval middleware can gate on a real predicate (see middleware.py).
  2. The eval suite gets an *oracle* — `evals/evaluators.py::policy_adherence`
     asserts the agent's stated decision matches what this function returned,
     which is far stronger evidence than asking a judge model if it looked right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from .config import (
    REFUND_AUTO_APPROVE_LIMIT,
    REFUND_HARD_CEILING,
    REFUND_WINDOW_DAYS,
)
from .db import money, query, query_one

Decision = Literal["auto_approve", "needs_human_approval", "deny"]


@dataclass
class RefundVerdict:
    """The complete, deterministic answer to 'can this order be refunded?'."""

    invoice_id: int
    decision: Decision
    reason_code: str
    reason: str
    refundable_amount: Decimal = Decimal("0.00")
    order_age_days: int | None = None
    order_total: Decimal = Decimal("0.00")
    #: True when a human needs to look at this *and the agent cannot proceed* —
    #: the supervisor reads this to route to the escalation specialist.
    requires_escalation: bool = False
    policy_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "invoice_id": self.invoice_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "refundable_amount": str(self.refundable_amount),
            "order_total": str(self.order_total),
            "order_age_days": self.order_age_days,
            "requires_escalation": self.requires_escalation,
            "policy_notes": self.policy_notes,
        }


def _parse_invoice_date(raw: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized invoice date format: {raw!r}")


def adjudicate(
    invoice_id: int,
    customer_id: int,
    *,
    now: datetime | None = None,
) -> RefundVerdict:
    """Decide the refund outcome for one order.

    Args:
        invoice_id: The order in question.
        customer_id: The *authenticated* caller. Passed so this function can
            re-verify ownership rather than trusting the caller to have done it.
        now: Injectable clock, so tests don't depend on the wall clock.

    Rules, applied in order — first match wins:
        1. Order not found, or not owned by this customer  -> deny
        2. Already refunded                                -> deny
        3. Older than REFUND_WINDOW_DAYS                   -> deny + escalate
        4. Above REFUND_HARD_CEILING                       -> deny + escalate
        5. Above REFUND_AUTO_APPROVE_LIMIT                 -> needs_human_approval
        6. Otherwise                                       -> auto_approve
    """
    now = now or datetime.now()

    invoice = query_one(
        "SELECT InvoiceId, CustomerId, InvoiceDate, Total FROM Invoice "
        "WHERE InvoiceId = ?",
        (invoice_id,),
    )

    # Rule 1. Note this deliberately returns the *same* message whether the order
    # doesn't exist or belongs to someone else. Distinguishing the two would leak
    # the existence of other customers' order ids.
    if invoice is None or invoice["CustomerId"] != customer_id:
        return RefundVerdict(
            invoice_id=invoice_id,
            decision="deny",
            reason_code="not_found",
            reason=(
                f"Order #{invoice_id} was not found on this account. Please "
                "double-check the order number."
            ),
        )

    total = money(invoice["Total"])
    age_days = (now - _parse_invoice_date(invoice["InvoiceDate"])).days

    # Rule 2
    prior = query_one(
        "SELECT RefundId, Amount, CreatedAt FROM Refund WHERE InvoiceId = ?",
        (invoice_id,),
    )
    if prior is not None:
        return RefundVerdict(
            invoice_id=invoice_id,
            decision="deny",
            reason_code="already_refunded",
            reason=(
                f"Order #{invoice_id} was already refunded "
                f"(${money(prior['Amount'])} on {prior['CreatedAt'][:10]})."
            ),
            order_total=total,
            order_age_days=age_days,
        )

    # Rule 3
    if age_days > REFUND_WINDOW_DAYS:
        return RefundVerdict(
            invoice_id=invoice_id,
            decision="deny",
            reason_code="outside_window",
            reason=(
                f"Order #{invoice_id} is {age_days} days old, outside the "
                f"{REFUND_WINDOW_DAYS}-day refund window."
            ),
            order_total=total,
            order_age_days=age_days,
            requires_escalation=True,
            policy_notes=[
                "A support representative can override the window as a goodwill "
                "gesture; the agent cannot."
            ],
        )

    # Rule 4
    if total > REFUND_HARD_CEILING:
        return RefundVerdict(
            invoice_id=invoice_id,
            decision="deny",
            reason_code="over_ceiling",
            reason=(
                f"Order #{invoice_id} totals ${total}, above the ${REFUND_HARD_CEILING} "
                "limit any automated process may refund."
            ),
            order_total=total,
            order_age_days=age_days,
            requires_escalation=True,
        )

    # Rule 5
    if total > REFUND_AUTO_APPROVE_LIMIT:
        return RefundVerdict(
            invoice_id=invoice_id,
            decision="needs_human_approval",
            reason_code="above_auto_limit",
            reason=(
                f"Order #{invoice_id} (${total}) is refundable but exceeds the "
                f"${REFUND_AUTO_APPROVE_LIMIT} auto-approval limit, so it needs "
                "sign-off from a support representative."
            ),
            refundable_amount=total,
            order_total=total,
            order_age_days=age_days,
        )

    # Rule 6
    return RefundVerdict(
        invoice_id=invoice_id,
        decision="auto_approve",
        reason_code="within_policy",
        reason=(
            f"Order #{invoice_id} (${total}) is {age_days} days old and within "
            "the refund policy. It can be refunded immediately."
        ),
        refundable_amount=total,
        order_total=total,
        order_age_days=age_days,
    )


def refund_history(customer_id: int) -> list[dict]:
    """Prior refunds for this customer — context for repeat-refunder patterns."""
    return query(
        "SELECT RefundId, InvoiceId, Amount, Reason, ApprovedBy, CreatedAt "
        "FROM Refund WHERE CustomerId = ? ORDER BY CreatedAt DESC",
        (customer_id,),
    )
