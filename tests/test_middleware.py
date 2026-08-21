"""What the customer reads when a safety rail fires.

A rail that stops the agent is only half a control. The other half is what the
person on the other end sees, and that half has no natural test — the rail fires
rarely, and when it does the run has already gone wrong, so nobody is looking
closely at the wording. The eval suite found the untested half: a run ended on the
literal string `Model call limits exceeded: run limit (4/4)`.
"""

from __future__ import annotations

import pytest
from langchain.agents.factory import _get_can_jump_to
from langchain.agents.middleware import HumanInTheLoopMiddleware

from chinook_support import middleware as mw
from chinook_support.config import GRAPH_RECURSION_LIMIT, MAX_ROUTING_HOPS
from chinook_support.graph import run_config
from chinook_support.middleware import BUDGET_EXCEEDED_REPLY, CallBudgetMiddleware
from chinook_support.security import AuditLogMiddleware


@pytest.fixture
def budget() -> CallBudgetMiddleware:
    return CallBudgetMiddleware(run_limit=3, exit_behavior="end")


def test_under_the_ceiling_it_does_nothing(budget):
    assert budget.before_model({"run_model_call_count": 2}, None) is None


def test_the_customer_never_reads_the_diagnostic(budget):
    result = budget.before_model({"run_model_call_count": 3}, None)

    assert result["jump_to"] == "end"
    (message,) = result["messages"]
    assert message.text == BUDGET_EXCEEDED_REPLY
    assert "limit" not in message.text.lower() or "Model call limits" not in message.text


def test_the_operator_still_can(budget):
    """The diagnostic moves off the reply, not out of the trace."""
    result = budget.before_model({"run_model_call_count": 3}, None)
    (message,) = result["messages"]

    diagnostic = message.additional_kwargs["call_budget_diagnostic"]
    assert "limit" in diagnostic.lower()
    assert "3" in diagnostic


def test_it_does_not_claim_the_work_finished(budget):
    """Some of the task may have landed and some may not.

    The run that exposed this had already filed the support case before the ceiling
    tripped. A reply asserting either "done" or "nothing happened" would be a guess,
    and guessing about whether money moved is its own incident.
    """
    reply = BUDGET_EXCEEDED_REPLY.lower()

    assert "not certain" in reply or "not sure" in reply
    for overclaim in ("all set", "completed", "nothing happened", "no changes"):
        assert overclaim not in reply


@pytest.mark.asyncio
async def test_the_async_path_substitutes_too(budget):
    """Studio invokes graphs asynchronously — see bug 1 in the README.

    `abefore_model` delegates to `before_model`, so the override covers both paths.
    That is inherited behaviour rather than anything this class does, which is
    exactly why it is worth a test: nothing here would notice if it changed.
    """
    result = await budget.abefore_model({"run_model_call_count": 3}, None)

    (message,) = result["messages"]
    assert message.text == BUDGET_EXCEEDED_REPLY


def test_the_jump_edge_is_still_wired(budget):
    """The substitution is worthless if the run doesn't actually end.

    `can_jump_to` is what makes the graph builder draw the conditional edge to
    `end`, and it is read off the overridden hook — overriding `before_model`
    without re-declaring it drops the metadata. The failure is silent in the worst
    way: `jump_to: "end"` with no edge to take means the ceiling stops nothing and
    the "I ran out of steps" line lands in the middle of a live conversation.
    """
    assert _get_can_jump_to(budget, "before_model") == ["end"]
    assert CallBudgetMiddleware.before_model.__can_jump_to__ == ["end"]


# --- Stack composition --------------------------------------------------------
#
# A stack is a list literal, so a control goes missing by omission rather than by
# breaking anything — nothing fails, the record just isn't written. These assert the
# claims the docstrings make about the stacks.


@pytest.mark.parametrize(
    "build",
    [mw.billing_middleware, mw.merch_middleware, mw.escalation_middleware],
    ids=["billing", "merch", "escalation"],
)
def test_every_specialist_is_audited(build):
    """`file_escalation` is a write path too.

    Escalation was the one stack without the audit middleware, so tickets got filed
    with no record of who asked for them — while `AuditLogMiddleware`'s own docstring
    named `file_escalation` as one of the three paths it covers.
    """
    assert any(isinstance(m, AuditLogMiddleware) for m in build())


def test_the_audit_record_says_which_area_wrote_it():
    (audit,) = [m for m in mw.escalation_middleware() if isinstance(m, AuditLogMiddleware)]
    assert audit.area == "escalation"


def _checkout_is_gated(stack) -> bool:
    """Is `checkout_cart` on the interrupt list?

    The middleware normalises its config and drops the tools configured `False`, so
    an ungated tool is an absent key rather than a falsy value.
    """
    (hitl,) = [m for m in stack if isinstance(m, HumanInTheLoopMiddleware)]
    return "checkout_cart" in hitl.interrupt_on


def test_checkout_is_gated():
    assert _checkout_is_gated(mw.merch_middleware())


def test_the_checkout_policy_constant_is_the_thing_that_decides(monkeypatch):
    """`CHECKOUT_ALWAYS_REQUIRES_APPROVAL` was a constant nothing imported.

    It read as the switch controlling checkout approval while the real gate was
    hard-coded in the stack, so editing it changed nothing. Flip it and the gate has
    to move, or it is decoration again.
    """
    monkeypatch.setattr(mw, "CHECKOUT_ALWAYS_REQUIRES_APPROVAL", False)
    assert not _checkout_is_gated(mw.merch_middleware())


# --- Superstep ceiling --------------------------------------------------------


def test_every_run_carries_a_recursion_limit():
    assert run_config("t")["recursion_limit"] == GRAPH_RECURSION_LIMIT
    assert run_config("t")["configurable"]["thread_id"] == "t"


def test_the_ceiling_clears_a_full_hop_budget():
    """It has to bound a runaway loop without tripping on a legitimate long turn.

    authenticate + MAX_ROUTING_HOPS x (supervisor + specialist) + a final supervisor
    + respond is the worst case the hop limit permits.
    """
    worst_case = 1 + 2 * MAX_ROUTING_HOPS + 1 + 1
    assert worst_case <= GRAPH_RECURSION_LIMIT < worst_case * 2
