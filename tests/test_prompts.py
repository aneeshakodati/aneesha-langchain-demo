"""Prompts, checked against the code they describe.

Prompts rot differently from code. Rename a tool and the call site fails loudly;
the sentence in the system prompt telling the model to call it by its old name
keeps "working", and the only symptom is a model that occasionally reaches for a
function that does not exist. Nothing in a normal test suite notices, because
nothing imports a prompt to check it.

So these tests treat the prompts as an interface description and assert it matches
the implementation:

  - every tool the prompts name exists;
  - every result field they tell the model to read is really returned;
  - every route label the router may emit is in `RouteDecision`;
  - the two rules the prompts exist to enforce (no policy numbers in prose, no
    echoing a third party's name) are still there.

They are cheap, offline, and they fail on the same commit that breaks them.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from chinook_support import prompts
from chinook_support.agents import RouteDecision
from chinook_support.cart import CartConstraints, CartPlan
from chinook_support.policy import RefundVerdict
from chinook_support.tools.billing import BILLING_TOOLS
from chinook_support.tools.escalation import ESCALATION_TOOLS
from chinook_support.tools.merch import MERCH_TOOLS

ALL_TOOLS = BILLING_TOOLS + MERCH_TOOLS + ESCALATION_TOOLS
TOOL_NAMES = {t.name for t in ALL_TOOLS}

CART_KEYS = set(
    CartPlan(items=[], total=Decimal("0.00"), constraints=CartConstraints()).to_dict()
)
VERDICT_KEYS = set(
    RefundVerdict(invoice_id=1, decision="deny", reason_code="x", reason="y").to_dict()
)
ROUTE_LABELS = set(RouteDecision.model_fields["next"].annotation.__args__)
#: `next`, `reason`, `task` — the router prompt instructs the model on how to fill
#: each of them, so it names them.
ROUTE_FIELDS = set(RouteDecision.model_fields)
REFUND_DECISIONS = {"auto_approve", "needs_human_approval", "deny"}

#: `file_escalation`'s own parameters, plus the classification vocabularies it
#: normalizes onto. The prompt is allowed to name any of them.
ESCALATION_VOCAB = {
    "refund_dispute", "billing_question", "order_problem", "catalog_request", "other",
    "low", "medium", "high", "calm", "frustrated", "angry",
}
TOOL_PARAMS = {
    name
    for tool in ALL_TOOLS
    for name in (tool.args_schema.model_fields if tool.args_schema else {})
}


def _all_prompts() -> dict[str, str]:
    """Every prompt string this package can send, including the rendered one."""
    found = {
        name: value
        for name, value in vars(prompts).items()
        if name.isupper() and isinstance(value, str)
    }
    found["respond_prompt"] = prompts.respond_prompt("Testcustomer")
    return found


def test_no_module_builds_a_system_message_from_a_literal():
    """`respond_prompt` used to be an f-string inline in `graph.py`, where no
    prompt review would ever have found it. The invariant that keeps it from
    happening again: a `SystemMessage(...)` anywhere in this package must be
    handed a name, not a string literal.
    """
    import ast
    from pathlib import Path

    offenders = []
    for path in Path(prompts.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "SystemMessage"
                and node.args
                and isinstance(node.args[0], (ast.Constant, ast.JoinedStr))
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"inline system prompt(s): {offenders}"


@pytest.mark.parametrize("name", sorted(_all_prompts()))
def test_every_identifier_a_prompt_names_actually_exists(name):
    """Backticked lowercase identifiers must be a real tool, result field, route
    label, or vocabulary value — not something the prompt invented."""
    allowed = (
        TOOL_NAMES
        | TOOL_PARAMS
        | CART_KEYS
        | VERDICT_KEYS
        | ROUTE_LABELS
        | ROUTE_FIELDS
        | REFUND_DECISIONS
        | ESCALATION_VOCAB
    )
    named = set(re.findall(r"`([a-z_][a-z0-9_]*)`", _all_prompts()[name]))
    assert named <= allowed, f"{name} names unknown identifiers: {sorted(named - allowed)}"


def test_the_billing_prompt_walks_the_real_decision_set():
    """Every decision `check_refund_eligibility` can return needs an instruction.
    A decision the prompt does not mention is one the model improvises around."""
    for decision in REFUND_DECISIONS:
        assert decision in prompts.BILLING_PROMPT


def test_the_router_prompt_covers_every_label_it_may_emit():
    for label in ROUTE_LABELS:
        assert f"`{label}`" in prompts.ROUTER_PROMPT


def test_no_prompt_states_a_policy_number():
    """The thresholds live in `config.py` and are enforced in `policy.py`. A prompt
    that also states them creates a second source of truth, and the two drift the
    first time someone edits one — so the billing prompt says "call the tool and
    relay what it returns" and never names a figure.

    Dollar amounts are allowed only inside an illustrative example of a
    `recommendation`, which is teaching the shape of a sentence, not a rule.
    """
    from chinook_support.config import (
        REFUND_AUTO_APPROVE_LIMIT,
        REFUND_HARD_CEILING,
        REFUND_WINDOW_DAYS,
    )

    for name, text in _all_prompts().items():
        for value in (REFUND_AUTO_APPROVE_LIMIT, REFUND_HARD_CEILING):
            assert f"${value}" not in text, f"{name} hardcodes a refund threshold"
        assert f"{REFUND_WINDOW_DAYS}-day" not in text, f"{name} hardcodes the window"
        assert f"{REFUND_WINDOW_DAYS} days" not in text, f"{name} hardcodes the window"


def test_the_store_voice_still_refuses_without_echoing_the_name():
    """Run 4 of the eval suite caught the agent refusing correctly and saying
    "Wyatt Girard" while doing it, which turns a refusal into a confirmation.
    The fix is a sentence in STORE_VOICE, so it is a sentence worth pinning."""
    voice = prompts.STORE_VOICE.lower()
    assert "that account" in voice
    assert "do not repeat" in voice or "should not" in voice


def test_every_specialist_prompt_carries_the_store_voice():
    """The tenant-isolation paragraph is in STORE_VOICE. A specialist prompt that
    forgot to prepend it would be the one place the agent has no such instruction."""
    for prompt in (
        prompts.BILLING_PROMPT,
        prompts.MERCH_PROMPT,
        prompts.ESCALATION_PROMPT,
        prompts.respond_prompt("Testcustomer"),
    ):
        assert prompts.STORE_VOICE in prompt


def test_the_respond_prompt_cannot_invent_account_facts():
    """This node answers without calling a single tool, so anything specific it
    says is by definition made up. That has to be stated, not implied."""
    text = prompts.respond_prompt("Luís")
    assert "Luís" in text
    assert "without using any tools" in text
    assert "invented" in text or "have not looked anything up" in text


def test_the_respond_prompt_advertises_the_human_handoff():
    """It answers "what can you help me with?", so its capability list is the
    product's public surface — and the handoff is what a stuck customer needs."""
    text = prompts.respond_prompt("Luís").lower()
    assert "representative" in text or "person" in text


def test_the_escalation_prompt_files_before_it_asks():
    """Run 4 caught "I'll get this straight to a person" with no ticket filed."""
    text = prompts.ESCALATION_PROMPT
    assert "File first, ask second" in text
    assert "file_escalation" in text


def test_the_escalation_prompt_decouples_severity_from_tone():
    """Run 5's remaining `calibrated` deduction: a ticket whose body flagged
    account probing filed itself as severity `medium`, sentiment `calm`. Politeness
    is not a triage signal."""
    text = prompts.ESCALATION_PROMPT.lower()
    assert "security" in text
    assert "high" in text
