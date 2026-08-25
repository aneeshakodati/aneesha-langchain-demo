"""Tenant isolation.

Every billing and escalation tool is invoked as each of two customers, and the
result is checked for the other one's data. These run without an LLM, so they're
fast and deterministic — the model's cooperation is not part of the control.
"""

from __future__ import annotations

import pytest
from langchain.agents.middleware import ToolCallRequest

from chinook_support.context import SupportContext, require_customer_id
from chinook_support.db import owns_invoice
from chinook_support.security import (
    CustomerScopeMiddleware,
    find_foreign_emails,
    _email_owners,
)
from chinook_support.tools.billing import (
    check_refund_eligibility,
    get_order_detail,
    list_my_orders,
)
from chinook_support.tools.escalation import get_my_support_rep

ALICE, BOB = 1, 2
BOBS_ORDER = 415


class FakeRuntime:
    """Minimal stand-in for ToolRuntime."""

    def __init__(self, customer_id: int | None):
        self.context = SupportContext(customer_id=customer_id)
        self.store = None
        self.state: dict = {}
        self.config: dict = {}
        self.tool_call_id = "test"
        self.stream_writer = None
        self.tools: list = []


def request_for(name: str, args: dict, customer_id: int | None = ALICE) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "t"},
        tool=None,
        state={},
        runtime=FakeRuntime(customer_id),
    )


# --- Tools are scoped by construction ----------------------------------------


def test_no_data_tool_accepts_a_customer_identifier():
    """The strongest control: the attack has no parameter to target."""
    for tool in (list_my_orders, get_order_detail, check_refund_eligibility,
                 get_my_support_rep):
        fields = set(tool.args_schema.model_fields)
        assert not fields & {"customer_id", "user_id", "account_id", "email"}, tool.name


def test_order_history_only_returns_the_callers_orders():
    for customer_id in (ALICE, BOB):
        result = list_my_orders.func(runtime=FakeRuntime(customer_id), limit=25)
        for order in result["orders"]:
            assert owns_invoice(customer_id, order["order_id"])


def test_order_detail_refuses_another_customers_order():
    result = get_order_detail.func(order_id=BOBS_ORDER, runtime=FakeRuntime(ALICE))
    assert result.get("error") == "not_found"
    assert "items" not in result


def test_order_detail_works_for_the_actual_owner():
    result = get_order_detail.func(order_id=BOBS_ORDER, runtime=FakeRuntime(BOB))
    assert result["order_id"] == BOBS_ORDER
    assert result["items"]


def test_unauthenticated_requests_are_refused():
    with pytest.raises(PermissionError):
        require_customer_id(SupportContext(customer_id=None))
    with pytest.raises(PermissionError):
        list_my_orders.func(runtime=FakeRuntime(None))


def test_no_tool_output_contains_another_customers_email():
    for customer_id in (ALICE, BOB):
        runtime = FakeRuntime(customer_id)
        payloads = [
            str(list_my_orders.func(runtime=runtime, limit=25)),
            str(get_my_support_rep.func(runtime=runtime)),
            str(check_refund_eligibility.func(order_id=413, runtime=runtime)),
        ]
        for payload in payloads:
            assert not find_foreign_emails(payload, customer_id)


def test_the_email_map_follows_the_active_database(tmp_path):
    """The leakage check must read the database the run is actually bound to.

    `_email_owners` was cached per process rather than per database, so whichever
    file was opened first supplied the email set for every later one. It failed
    *open*: an address the stale map had never seen was not a known customer's, so
    nothing was reported. The eval harness gives every example a private copy, and
    this is the one control where being right about the wrong database is silent.
    """
    import shutil
    import sqlite3

    from chinook_support.config import DEMO_DB
    from chinook_support.db import use_db

    other = tmp_path / "other.db"
    shutil.copyfile(DEMO_DB, other)
    connection = sqlite3.connect(other)
    connection.execute(
        "UPDATE Customer SET Email = ? WHERE CustomerId = ?", ("elsewhere@example.com", BOB)
    )
    connection.commit()
    connection.close()

    # Populate the cache from the default database first — that is the ordering
    # that produced the bug.
    _email_owners()

    with use_db(other):
        assert find_foreign_emails("contact them at elsewhere@example.com", ALICE) == [
            "elsewhere@example.com"
        ]
        # And the caller's own address is still not a leak.
        assert not find_foreign_emails("contact them at elsewhere@example.com", BOB)


# --- The middleware backstop --------------------------------------------------


def test_guard_rejects_a_forbidden_argument_before_the_tool_runs():
    def must_not_run(_request):
        raise AssertionError("tool executed despite a forbidden argument")

    blocked = CustomerScopeMiddleware().wrap_tool_call(
        request_for("list_my_orders", {"customer_id": 42}), must_not_run
    )
    assert blocked.status == "error"
    assert "customer_id" in blocked.content


@pytest.mark.parametrize(
    "args", [{"user_id": 3}, {"account_id": 9}, {"on_behalf_of": "bob@example.com"}]
)
def test_guard_rejects_every_identity_bearing_argument(args):
    blocked = CustomerScopeMiddleware().wrap_tool_call(
        request_for("list_my_orders", args),
        lambda _r: pytest.fail("should not execute"),
    )
    assert blocked.status == "error"


def test_guard_withholds_a_leaky_tool_result():
    """Simulates a future tool that accidentally returns someone else's record."""
    foreign_email = next(
        email for email, owner in _email_owners().items() if owner != ALICE
    )

    class Result:
        content = f"Here you go: {foreign_email}"
        status = "success"

    blocked = CustomerScopeMiddleware().wrap_tool_call(
        request_for("list_my_orders", {}), lambda _r: Result()
    )
    assert blocked.status == "error"
    assert foreign_email not in blocked.content


def test_guard_passes_clean_results_through_untouched():
    class Result:
        content = "Order #413, $5.94"
        status = "success"

    result = Result()
    assert CustomerScopeMiddleware().wrap_tool_call(
        request_for("list_my_orders", {}), lambda _r: result
    ) is result


@pytest.mark.asyncio
async def test_the_async_path_enforces_the_same_rules():
    """Studio and LangGraph Platform run graphs async.

    A guard implemented only for the sync path passes every local test and is
    absent in the deployment that matters.
    """
    guard = CustomerScopeMiddleware()

    async def must_not_run(_request):
        raise AssertionError("tool executed despite a forbidden argument")

    blocked = await guard.awrap_tool_call(
        request_for("list_my_orders", {"customer_id": 42}), must_not_run
    )
    assert blocked.status == "error"

    async def leaky(_request):
        foreign = next(e for e, owner in _email_owners().items() if owner != ALICE)
        return type("M", (), {"content": foreign, "status": "success"})()

    withheld = await guard.awrap_tool_call(request_for("list_my_orders", {}), leaky)
    assert withheld.status == "error"
