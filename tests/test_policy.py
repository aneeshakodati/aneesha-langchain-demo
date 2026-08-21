"""Refund policy engine.

These are the tests that matter most in the project. The policy engine is the
oracle the eval suite grades the agent against, so if it's wrong, the evals
confidently certify wrong behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from chinook_support.config import REFUND_AUTO_APPROVE_LIMIT, REFUND_WINDOW_DAYS
from chinook_support.db import query_one, write_conn
from chinook_support.policy import adjudicate

# Seeded by scripts/build_db.py -- see SEED_ORDERS there.
RECENT_SMALL = 413  # ~$5.94, 5 days old, customer 1
RECENT_LARGE = 414  # ~$25.74, 12 days old, customer 1
OTHER_CUSTOMER = 415  # customer 2
STALE = 416  # 200 days old, customer 3


def test_small_recent_order_auto_approves():
    verdict = adjudicate(RECENT_SMALL, customer_id=1)
    assert verdict.decision == "auto_approve"
    assert verdict.reason_code == "within_policy"
    assert verdict.refundable_amount <= REFUND_AUTO_APPROVE_LIMIT
    assert not verdict.requires_escalation


def test_large_recent_order_needs_a_human():
    verdict = adjudicate(RECENT_LARGE, customer_id=1)
    assert verdict.decision == "needs_human_approval"
    assert verdict.refundable_amount > REFUND_AUTO_APPROVE_LIMIT
    # Needing approval is not the same as needing escalation: a representative
    # approves it inline, the case is not handed off.
    assert not verdict.requires_escalation


def test_order_outside_the_window_is_denied_and_escalated():
    verdict = adjudicate(STALE, customer_id=3)
    assert verdict.decision == "deny"
    assert verdict.reason_code == "outside_window"
    assert verdict.order_age_days > REFUND_WINDOW_DAYS
    assert verdict.requires_escalation


def test_another_customers_order_is_indistinguishable_from_a_missing_one():
    """The denial must not reveal that the order exists.

    If 'belongs to someone else' and 'does not exist' produced different
    responses, the agent would be an oracle for probing valid order ids.
    """
    foreign = adjudicate(OTHER_CUSTOMER, customer_id=1)
    missing = adjudicate(9_999_999, customer_id=1)

    assert foreign.decision == missing.decision == "deny"
    assert foreign.reason_code == missing.reason_code == "not_found"
    assert foreign.reason.replace(str(OTHER_CUSTOMER), "X") == missing.reason.replace(
        "9999999", "X"
    )
    # And it must not disclose the real total.
    assert foreign.order_total == Decimal("0.00")


def test_the_same_order_is_refundable_for_its_actual_owner():
    assert adjudicate(OTHER_CUSTOMER, customer_id=2).decision != "deny"


def test_window_boundary_is_inclusive():
    """Exactly at the window is inside it; one day past is not."""
    invoice = query_one("SELECT InvoiceDate FROM Invoice WHERE InvoiceId = ?", (RECENT_SMALL,))
    placed = datetime.strptime(invoice["InvoiceDate"][:19], "%Y-%m-%d %H:%M:%S")

    on_boundary = placed + timedelta(days=REFUND_WINDOW_DAYS)
    assert adjudicate(RECENT_SMALL, 1, now=on_boundary).decision == "auto_approve"

    past = placed + timedelta(days=REFUND_WINDOW_DAYS + 1)
    assert adjudicate(RECENT_SMALL, 1, now=past).reason_code == "outside_window"


def test_an_already_refunded_order_cannot_be_refunded_twice():
    with write_conn() as conn:
        conn.execute(
            "INSERT INTO Refund (InvoiceId, CustomerId, Amount, Reason, ApprovedBy, "
            "CreatedAt) VALUES (?, 1, 1.00, 'test', 'policy:auto', ?)",
            (RECENT_SMALL, datetime.now().isoformat(sep=" ", timespec="seconds")),
        )
    try:
        verdict = adjudicate(RECENT_SMALL, customer_id=1)
        assert verdict.decision == "deny"
        assert verdict.reason_code == "already_refunded"
    finally:
        with write_conn() as conn:
            conn.execute("DELETE FROM Refund WHERE InvoiceId = ? AND Reason = 'test'",
                         (RECENT_SMALL,))


@pytest.mark.parametrize("invoice_id", [RECENT_SMALL, RECENT_LARGE, STALE])
def test_adjudication_is_deterministic(invoice_id):
    """Same inputs, same verdict -- this is what makes it usable as an eval oracle."""
    now = datetime(2026, 8, 20, 12, 0, 0)
    first = adjudicate(invoice_id, customer_id=1, now=now)
    second = adjudicate(invoice_id, customer_id=1, now=now)
    assert first.to_dict() == second.to_dict()
