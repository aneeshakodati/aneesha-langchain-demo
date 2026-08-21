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

from chinook_support.middleware import BUDGET_EXCEEDED_REPLY, CallBudgetMiddleware


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
