"""Escalation tools — hand off to the customer's assigned support representative.

Chinook already models this: every customer has a `SupportRepId` pointing at an
Employee. So escalation isn't invented for the demo, it's the store's own
org chart — Jane, Margaret, and Steve each carry about twenty accounts.

The interesting part is the *summary*. A handoff that says "customer wants a
refund, please help" wastes the representative's time; they have to re-read the
whole transcript. So `file_escalation`'s signature is the ticket schema: the
model cannot file a case without separately stating what the customer wants, what
was already tried, and what it recommends the representative actually do.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain.tools import ToolRuntime, tool

from ..context import require_customer_id
from ..db import get_customer, query, write_conn


@tool
def get_my_support_rep(runtime: ToolRuntime) -> dict:
    """Look up which support representative is assigned to this customer.

    Call this before filing an escalation so the ticket is routed to the right
    person and you can tell the customer who will follow up.
    """
    customer_id = require_customer_id(runtime.context)
    customer = get_customer(customer_id)
    if customer is None or customer["SupportRepId"] is None:
        return {
            "assigned": False,
            "message": "No representative is assigned; the ticket will go to the general queue.",
        }
    return {
        "assigned": True,
        "rep_id": customer["SupportRepId"],
        "rep_name": f"{customer['RepFirstName']} {customer['RepLastName']}",
        "rep_title": customer["RepTitle"],
    }


# Classification vocabularies, with synonyms.
#
# These were `Literal[...]` annotations, which made them strict enum validation on
# the tool schema. That is the textbook choice and it was wrong here. A customer
# said they were "annoyed", the model passed `sentiment="annoyed"`, pydantic
# rejected it, the retry sent the identical arguments, and the escalation was
# never filed — so the one customer in the demo who most needed a human got told
# "I'm hitting a technical error" instead.
#
# The trade is asymmetric. A slightly-off severity label costs a representative
# nothing; a dropped ticket for an angry customer is the worst outcome the system
# can produce. So these accept free text and normalize, and the case always files.
# The label the model actually used is preserved in the result so nothing is
# silently rewritten.
_CATEGORIES = {
    "refund_dispute": {"refund", "refund_request", "refund_dispute", "chargeback", "dispute"},
    "billing_question": {"billing", "billing_question", "charge", "payment", "invoice"},
    "order_problem": {"order_problem", "order", "delivery", "download", "playback", "defect"},
    "catalog_request": {"catalog_request", "catalog", "request", "availability", "stock"},
    "other": {"other", "general", "misc"},
}
_SEVERITIES = {
    "low": {"low", "minor", "trivial", "informational"},
    "medium": {"medium", "normal", "moderate", "standard"},
    "high": {"high", "urgent", "critical", "blocker", "severe", "escalated"},
}
_SENTIMENTS = {
    "calm": {"calm", "neutral", "positive", "happy", "polite", "patient", "satisfied"},
    "frustrated": {
        "frustrated", "annoyed", "irritated", "upset", "disappointed", "unhappy",
        "impatient", "concerned", "confused",
    },
    "angry": {"angry", "furious", "irate", "livid", "hostile", "outraged"},
}


def _normalize(value: str, vocabulary: dict[str, set[str]], default: str) -> str:
    """Map free text onto a known label, falling back rather than failing."""
    token = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    for label, synonyms in vocabulary.items():
        if token == label or token in synonyms:
            return label
    # Last resort: substring match, so "very_frustrated" still lands somewhere sane.
    for label, synonyms in vocabulary.items():
        if any(synonym in token for synonym in synonyms):
            return label
    return default


@tool
def file_escalation(
    runtime: ToolRuntime,
    category: str,
    severity: str,
    sentiment: str,
    subject: str,
    summary: str,
    steps_taken: str,
    recommendation: str,
    related_order_ids: list[int] | None = None,
) -> dict:
    """File a support case for a human representative to pick up.

    Call this once, at the end, after you have gathered the facts. Write the
    summary for a representative who has not read the conversation: what the
    customer wants, what is actually true about their account, and what you
    already tried. Do not include the customer's email or payment details.

    Args:
        category: One of refund_dispute, billing_question, order_problem,
            catalog_request, other.
        severity: One of low, medium, high. Use high for anything blocking or
            involving money the customer has already paid.
        sentiment: One of calm, frustrated, angry — how the customer sounds, so
            the representative can pitch their reply.
        subject: One-line case title.
        summary: What the customer wants and the relevant account facts.
        steps_taken: What the agent already checked or attempted, and outcomes.
        recommendation: The concrete next action you suggest the rep take.
        related_order_ids: Order numbers this case concerns.
    """
    customer_id = require_customer_id(runtime.context)

    raw = {"category": category, "severity": severity, "sentiment": sentiment}
    category = _normalize(category, _CATEGORIES, "other")
    severity = _normalize(severity, _SEVERITIES, "medium")
    sentiment = _normalize(sentiment, _SENTIMENTS, "frustrated")
    normalized = {
        field: {"given": given, "stored": value}
        for field, given, value in (
            ("category", raw["category"], category),
            ("severity", raw["severity"], severity),
            ("sentiment", raw["sentiment"], sentiment),
        )
        if (given or "").strip().lower() != value
    }
    customer = get_customer(customer_id)
    rep_id = customer["SupportRepId"] if customer else None

    # Only file against orders the customer actually owns, so a malformed or
    # injected order number can't attach someone else's invoice to this case.
    owned: list[int] = []
    if related_order_ids:
        rows = query(
            "SELECT InvoiceId FROM Invoice WHERE CustomerId = ? AND InvoiceId IN "
            f"({','.join('?' * len(related_order_ids))})",
            (customer_id, *[int(o) for o in related_order_ids]),
        )
        owned = [r["InvoiceId"] for r in rows]

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with write_conn() as conn:
        cur = conn.execute(
            "INSERT INTO SupportCase (CustomerId, AssignedRepId, Category, Severity, "
            "Sentiment, Subject, Summary, StepsTaken, Recommendation, "
            "RelatedInvoices, Status, CreatedAt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
            (
                customer_id,
                rep_id,
                category,
                severity,
                sentiment,
                subject,
                summary,
                steps_taken,
                recommendation,
                json.dumps(owned),
                now,
            ),
        )
        case_id = cur.lastrowid

    rep_name = (
        f"{customer['RepFirstName']} {customer['RepLastName']}"
        if customer and customer["RepFirstName"]
        else "the support team"
    )
    result = {
        "filed": True,
        "case_id": case_id,
        "assigned_to": rep_name,
        "category": category,
        "severity": severity,
        "sentiment": sentiment,
        "related_order_ids": owned,
        "message": (
            f"Case #{case_id} has been opened and assigned to {rep_name}. "
            "They typically respond within one business day."
        ),
    }
    if normalized:
        result["normalized_fields"] = normalized
    return result


ESCALATION_TOOLS = [get_my_support_rep, file_escalation]
