"""Middleware stacks, assembled per specialist area.

The stacks are deliberately different from each other, and that difference is the
argument for splitting the agent into specialists in the first place. Billing moves
money, so it gets human approval and an audit trail. Merch runs long browsing
sessions, so it gets a search budget and conversation summarization. Escalation
emits a record that leaves the system, so it gets PII redaction.

A single flat agent would have to carry the union of all of it on every request:
paying the summarization check on a one-line refund question, and running the
approval machinery while someone browses jazz.
"""

from __future__ import annotations

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolRetryMiddleware,
    dynamic_prompt,
    hook_config,
)
from langchain_core.messages import AIMessage

from .config import (
    MAX_CATALOG_SEARCHES_PER_RUN,
    MAX_MODEL_CALLS_ESCALATION,
    MAX_MODEL_CALLS_PER_RUN,
    SUMMARY_MODEL,
)
from .context import coerce_context
from .db import get_customer
from .policy import adjudicate
from .security import AuditLogMiddleware, CustomerScopeMiddleware

# --- Call ceiling -------------------------------------------------------------

#: What the customer reads when an agent runs out of model calls mid-task.
#:
#: Deliberately non-committal about what got done. When the ceiling trips, some of
#: the work may have landed and some may not — the run the eval suite caught had
#: already filed the support case and then died before it could say so. Claiming
#: either "that's done" or "nothing happened" would be a guess, and a support bot
#: guessing about whether it refunded you is its own incident.
BUDGET_EXCEEDED_REPLY = (
    "Sorry — I ran out of steps on this one before I could wrap it up, so I'm not "
    "certain everything went through. Ask me again and I'll pick it back up, or say "
    "\"get me a person\" and I'll pass it to your support representative."
)


class CallBudgetMiddleware(ModelCallLimitMiddleware):
    """`ModelCallLimitMiddleware`, except the customer doesn't read the diagnostic.

    The stock middleware ends the run by injecting `Model call limits exceeded: run
    limit (4/4)` as an assistant message. That is the right text for an operator and
    the wrong text for the person who asked about their order — and because it is a
    perfectly good AI message with content in it, every downstream "did anyone
    answer?" check is satisfied by it, so it sails through as the final reply. The
    eval suite caught exactly that.

    The diagnostic is not thrown away; it moves to `additional_kwargs`, so it is
    still on the message in the LangSmith trace where an operator will look for it.

    The `hook_config` decorator has to be repeated here. It is what tells the graph
    builder to wire a conditional edge to `end`, and it is read off the hook that is
    actually overridden — inheriting the behaviour does not inherit the metadata.
    Without it the wiring survives only because the *inherited async* hook still
    carries the decorator, which is the same sync/async asymmetry that hid bug 1 in
    the README, and it would fail silently: `jump_to: "end"` with no edge to take
    means the ceiling stops nothing and the substituted reply lands mid-conversation.
    """

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):  # type: ignore[no-untyped-def]
        result = super().before_model(state, runtime)
        if not result or result.get("jump_to") != "end":
            return result
        diagnostic = " ".join(m.text for m in result.get("messages", []))
        return {
            **result,
            "messages": [
                AIMessage(
                    content=BUDGET_EXCEEDED_REPLY,
                    additional_kwargs={"call_budget_diagnostic": diagnostic},
                )
            ],
        }


# --- Human-in-the-loop predicates --------------------------------------------
#
# `InterruptOnConfig.when` receives the actual ToolCallRequest, so the decision to
# involve a human is made by running the real policy engine against the real
# arguments -- not by the model deciding whether it feels risky.


def refund_needs_human(request: ToolCallRequest) -> bool:
    """Interrupt only for refunds the policy engine says need sign-off.

    A $4 refund inside the window goes straight through; nobody wants to page a
    representative for that. A $25.74 refund stops here. If the engine denies the
    refund outright there is nothing to approve, so we let the tool run and return
    its denial rather than asking a human to rubber-stamp a "no".
    """
    context = coerce_context(getattr(request.runtime, "context", None))
    if context.customer_id is None:
        return True  # fail closed

    order_id = (request.tool_call.get("args") or {}).get("order_id")
    if order_id is None:
        return True

    try:
        verdict = adjudicate(int(order_id), context.customer_id)
    except (ValueError, TypeError):
        return True

    return verdict.decision == "needs_human_approval"


# Note the signature difference: `when` receives a ToolCallRequest, while
# `description` receives (tool_call, state, runtime). Easy to get wrong; the
# failure is a TypeError at interrupt time, i.e. exactly when you least want one.


def describe_refund(tool_call: dict, state, runtime) -> str:
    """Give the approver the facts, not just the raw tool call.

    An approval prompt that shows `issue_refund({"order_id": 414, ...})` asks a
    human to rubber-stamp something they'd have to go look up. Show them the
    amount, the age, and the customer's stated reason so the decision is real.
    """
    context = coerce_context(getattr(runtime, "context", None))
    args = tool_call.get("args") or {}
    order_id = args.get("order_id")
    reason = args.get("reason", "(no reason given)")
    try:
        verdict = adjudicate(int(order_id), context.customer_id)
        amount, age = f"${verdict.refundable_amount}", f"{verdict.order_age_days} days"
    except Exception:  # noqa: BLE001
        amount, age = "unknown", "unknown"
    return (
        f"REFUND APPROVAL NEEDED\n"
        f"  Customer : #{context.customer_id}\n"
        f"  Order    : #{order_id}\n"
        f"  Amount   : {amount}\n"
        f"  Age      : {age}\n"
        f"  Reason   : {reason}\n"
        f"Approve to issue the refund, or reject with feedback."
    )


def describe_checkout(tool_call: dict, state, runtime) -> str:
    context = coerce_context(getattr(runtime, "context", None))
    return (
        f"ORDER CONFIRMATION NEEDED\n"
        f"  Customer: #{context.customer_id}\n"
        f"Checkout will create a real order and charge the cart total.\n"
        f"Approve to place it, or reject with feedback."
    )


# --- Dynamic prompt ----------------------------------------------------------


def with_customer_profile(base_prompt: str) -> AgentMiddleware:
    """Build a middleware that appends the authenticated customer to the prompt.

    Doing this at request time rather than baking a name into a static prompt
    means the prompt is structurally incapable of being about the wrong person:
    it is rendered from the same runtime context the tools are scoped by.
    """

    @dynamic_prompt
    def _prompt(request: ModelRequest) -> str:
        context = coerce_context(getattr(request.runtime, "context", None))
        if context.customer_id is None:
            return (
                base_prompt
                + "\n\nThe caller is NOT authenticated. Do not use any tool. Tell "
                "them they need to sign in."
            )

        customer = get_customer(context.customer_id)
        if customer is None:
            return base_prompt

        rep = (
            f"{customer['RepFirstName']} {customer['RepLastName']}"
            if customer["RepFirstName"]
            else "unassigned"
        )
        location = ", ".join(
            part for part in (customer["City"], customer["Country"]) if part
        )
        acting = (
            f"\nA staff member ({context.staff_agent_email}) is driving this "
            "conversation on the customer's behalf. Data access is unchanged."
            if context.staff_agent_email
            else ""
        )
        return (
            f"{base_prompt}\n\n"
            f"--- Authenticated customer (from the session, not from the chat) ---\n"
            f"Name    : {customer['FirstName']} {customer['LastName']}\n"
            f"Account : #{customer['CustomerId']}\n"
            f"Location: {location}\n"
            f"Rep     : {rep}\n"
            f"Channel : {context.channel}{acting}\n"
            f"Address them by first name. Every tool you call is already scoped to "
            f"this account."
        )

    return _prompt


# --- Stacks ------------------------------------------------------------------
#
# Order matters. CustomerScopeMiddleware is first so its pre-execution check runs
# before anything else touches the call, and its post-execution scan runs last on
# the way back out.


def billing_middleware() -> list[AgentMiddleware]:
    from .prompts import BILLING_PROMPT

    return [
        CustomerScopeMiddleware(),
        AuditLogMiddleware(area="billing"),
        with_customer_profile(BILLING_PROMPT),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": refund_needs_human,
                    "description": describe_refund,
                },
                # Reads are never gated. Spelling them out as False documents the
                # decision rather than leaving it to the default.
                "list_my_orders": False,
                "get_order_detail": False,
                "check_refund_eligibility": False,
            }
        ),
        ToolRetryMiddleware(max_retries=2, initial_delay=0.5),
        CallBudgetMiddleware(run_limit=MAX_MODEL_CALLS_PER_RUN, exit_behavior="end"),
    ]


def merch_middleware() -> list[AgentMiddleware]:
    from .prompts import MERCH_PROMPT

    return [
        CustomerScopeMiddleware(),
        AuditLogMiddleware(area="merch"),
        with_customer_profile(MERCH_PROMPT),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "checkout_cart": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": describe_checkout,
                },
                "search_catalog": False,
                "build_music_cart": False,
                "view_cart": False,
                "add_tracks_to_cart": False,
                "remove_tracks_from_cart": False,
            }
        ),
        # Browsing is the one flow that can genuinely run away: each search returns
        # 25 rows and the model always wants one more. Cap it.
        ToolCallLimitMiddleware(
            tool_name="search_catalog",
            run_limit=MAX_CATALOG_SEARCHES_PER_RUN,
            exit_behavior="continue",
        ),
        # Long browse sessions are the only place context actually gets tight.
        SummarizationMiddleware(
            model=SUMMARY_MODEL,
            trigger=("tokens", 60_000),
            keep=("messages", 12),
        ),
        ToolRetryMiddleware(max_retries=2, initial_delay=0.5),
        CallBudgetMiddleware(run_limit=MAX_MODEL_CALLS_PER_RUN, exit_behavior="end"),
    ]


def escalation_middleware() -> list[AgentMiddleware]:
    from .prompts import ESCALATION_PROMPT

    return [
        CustomerScopeMiddleware(),
        with_customer_profile(ESCALATION_PROMPT),
        # A support case is read by a human and may be exported to a ticketing
        # system, so scrub contact details on the way out even though the prompt
        # already says not to include them. Prompts are not a control.
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        PIIMiddleware("credit_card", strategy="redact", apply_to_output=True),
        ToolRetryMiddleware(max_retries=2, initial_delay=0.5),
        CallBudgetMiddleware(run_limit=MAX_MODEL_CALLS_ESCALATION, exit_behavior="end"),
    ]
