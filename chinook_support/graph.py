"""The parent graph: authenticate -> supervisor -> specialist -> supervisor -> end.

    START
      |
      v
 [authenticate]   resolve the caller from runtime context; refuse if absent
      |
      v
 [supervisor] <---------------------+   structured-output router, hop-limited
      |  |  |  |                    |
      |  |  |  +--> [escalation] ---+
      |  |  +-----> [billing] ------+
      |  +--------> [merch] --------+
      |
      +-----------> [respond] --> END   direct reply when no specialist is needed

Why a real StateGraph instead of wrapping the specialists as tools on one agent:

- It is legible. In LangGraph Studio you watch control move through the nodes, so
  the architecture explains itself to anyone watching the demo.
- The specialists share `messages`, so the escalation agent can see everything
  billing already tried. A handoff summary is only good if the writer has the
  history, and subagent-as-tool would hand it a one-line task description.
- Routing becomes an addressable, evaluable step rather than an implementation
  detail buried inside a tool call.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from typing_extensions import Annotated, NotRequired, TypedDict

from .agents import (
    RouteDecision,
    build_billing_agent,
    build_escalation_agent,
    build_merch_agent,
)
from .config import GRAPH_RECURSION_LIMIT, MAX_ROUTING_HOPS, router_model
from .context import SupportContext, coerce_context
from .db import get_customer
from .prompts import ROUTER_PROMPT, respond_prompt

SPECIALISTS = ("billing", "merch", "escalation")

#: `name` stamped on the supervisor's internal handoff messages. They are real
#: messages in the thread (the specialists need them), but they are not the
#: customer talking, so anything user-facing should filter them out —
#: see `visible_messages`.
HANDOFF_SENDER = "supervisor"


class SupportState(TypedDict):
    """Shared state. `messages` is the only thing the specialists read or write."""

    messages: Annotated[list[AnyMessage], add_messages]
    #: Where the supervisor sent us last. Surfaced for tracing and eval.
    route: NotRequired[str]
    route_reason: NotRequired[str]
    #: Supervisor -> specialist -> supervisor round trips this turn. Overwrite
    #: semantics (no reducer) so `authenticate` can reset it every turn.
    hops: NotRequired[int]
    authenticated: NotRequired[bool]
    customer_name: NotRequired[str]


def authenticate(state: SupportState, runtime: Runtime[SupportContext]) -> dict:
    """Resolve the caller from runtime context. First node, every turn.

    This is the only place identity is established, and it reads from runtime
    context rather than from the conversation. Nothing the customer types reaches
    this function.
    """
    context = coerce_context(runtime.context)

    if context.customer_id is None:
        return {
            "authenticated": False,
            "hops": 0,
            "route": "finish",
            "messages": [
                AIMessage(
                    "I can't access any account information until you're signed in. "
                    "Please sign in and I'll pick up right where we left off."
                )
            ],
        }

    customer = get_customer(context.customer_id)
    if customer is None:
        return {
            "authenticated": False,
            "hops": 0,
            "route": "finish",
            "messages": [
                AIMessage(
                    "I couldn't load your account just now. Please try again, or "
                    "contact support@chinookcorp.com if it keeps happening."
                )
            ],
        }

    return {
        "authenticated": True,
        "hops": 0,
        "customer_name": f"{customer['FirstName']} {customer['LastName']}",
    }


#: Node name -> the route label the supervisor uses for it.
_AREA_OF_AGENT = {
    "billing_agent": "billing",
    "merch_agent": "merch",
    "escalation_agent": "escalation",
}


def text_of(message: AnyMessage) -> str:
    """The human-readable text of a message, and nothing else.

    Anthropic returns *content blocks* — a list — whenever extended thinking is on,
    so `message.content` is `[{"type": "thinking", "signature": "EocQ...base64..."},
    {"type": "text", "text": "..."}]`. Anything that did `str(message.content)`
    therefore got a stringified list with an encrypted thinking blob in it. That
    reached three places that matter: the router's transcript (so the cheap model
    read base64 instead of the conversation), the "has a specialist already
    answered?" check, and the reply the eval suite graded. `.text` concatenates the
    text blocks and drops the rest, which is what every one of those wanted.
    """
    return (message.text or "").strip()


def _last_specialist_area(messages: list[AnyMessage]) -> str | None:
    """Which specialist produced the final answer, if the customer hasn't replied.

    Returns None as soon as a customer message is more recent than any specialist
    reply, because at that point every specialist is fair game again.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not is_handoff(message):
            return None
        if isinstance(message, AIMessage) and text_of(message) and not message.tool_calls:
            return _AREA_OF_AGENT.get(getattr(message, "name", "") or "")
    return None


def is_handoff(message: AnyMessage) -> bool:
    """True for the supervisor's internal routing messages."""
    return getattr(message, "name", None) == HANDOFF_SENDER


def visible_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop internal handoffs so a UI shows only the real conversation."""
    return [m for m in messages if not is_handoff(m)]


def _transcript(messages: list[AnyMessage], limit: int = 8) -> str:
    """Compact recent history for the router.

    The router is a cheap model making one narrow decision; handing it the full
    message objects (with tool calls and JSON payloads) costs tokens and makes the
    decision harder, not easier.
    """
    lines = []
    for message in messages[-limit:]:
        role = getattr(message, "type", "?")
        if role == "tool":
            continue
        text = text_of(message)
        if not text:
            # An assistant turn that was purely tool calls.
            calls = getattr(message, "tool_calls", None)
            if calls:
                text = f"(called tools: {', '.join(c['name'] for c in calls)})"
            else:
                continue
        speaker = {"human": "Customer", "ai": "Assistant", "system": "System"}.get(role, role)
        if is_handoff(message):
            speaker = "Supervisor"
        lines.append(f"{speaker}: {text[:600]}")
    return "\n".join(lines)


#: What the customer reads when the model itself is unreachable.
UNREACHABLE_REPLY = (
    "Sorry — I'm having trouble reaching my systems right now. Please try that "
    "again in a moment, and if it keeps happening, email support@chinookcorp.com "
    "and a person will pick it up."
)


def _router_model(structured: type | None = None):
    """The router model, with a retry around it. Built once per (model, schema).

    The specialists get `ToolRetryMiddleware`, but these two nodes call the model
    directly and had nothing around them at all — so a single transient
    `AnthropicConnectionError` killed the whole turn. That is not hypothetical: it
    took down `make demo` mid-run. Every model call in this graph is now retried,
    because the one that isn't is the one that fails in front of an audience.

    Three attempts with exponential jitter. Retrying on `Exception` rather than a
    curated list of provider error classes is deliberate — the failure mode of
    retrying something unretryable is a few wasted seconds, and the failure mode of
    a missing entry in the list is a dead conversation.

    Retry lives on the model rather than on the node (`add_node(retry_policy=...)`)
    on purpose. A node retry re-runs the node from the top, and for a node that has
    already had side effects that means doing them twice. Here that would be
    harmless — the supervisor only reads — but keeping every retry in this graph at
    the model layer means the rule is one rule, and it stays true when someone gives
    the supervisor something to write.

    The cache is keyed on the resolved model id as well as the schema, so a
    `ROUTER_MODEL` override still takes effect (see the note in config.py) instead
    of being frozen by whichever value happened to be read first.
    """
    return _build_router_model(router_model(), structured)


@lru_cache(maxsize=None)
def _build_router_model(model_id: str, structured: type | None):
    model = init_chat_model(model_id)
    if structured is not None:
        model = model.with_structured_output(structured)
    return model.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def supervisor(state: SupportState, runtime: Runtime[SupportContext]) -> dict:
    """Choose the next specialist, or stop.

    Hard-stops at MAX_ROUTING_HOPS. Without a ceiling, a router and a specialist
    that disagree about whether something is resolved will ping-pong forever, and
    the customer watches a spinner while the bill runs.
    """
    if not state.get("authenticated", False):
        return {"route": "finish", "route_reason": "not authenticated"}

    hops = state.get("hops", 0)
    if hops >= MAX_ROUTING_HOPS:
        return {
            "route": "finish",
            "route_reason": f"hop limit ({MAX_ROUTING_HOPS}) reached",
            "hops": hops,
        }

    try:
        decision: RouteDecision = _router_model(RouteDecision).invoke(
            [
                SystemMessage(ROUTER_PROMPT),
                HumanMessage(
                    f"Conversation so far:\n\n{_transcript(state['messages'])}\n\n"
                    f"Specialists already used this turn: {hops}. Who acts next?"
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        # Retries are exhausted. Fall through to `respond` rather than raising:
        # routing is the one step with a safe default, and a customer who gets an
        # apology is in a better place than one whose turn vanished into a stack
        # trace. The reason lands in state, so the trace still says what happened.
        return {
            "route": "finish",
            "route_reason": f"router unavailable ({type(exc).__name__}); falling back",
            "hops": hops + 1,
        }

    next_area = decision.next

    # Deterministic guardrail over the model's choice. Left to itself the router
    # will occasionally send the conversation back to the specialist that just
    # answered; that specialist has nothing left to do, so it emits filler like
    # "let me know if you need anything else" and the customer sees two replies.
    # A specialist never follows itself without the customer saying something in
    # between -- that's a rule, so it's enforced here rather than asked for in the
    # prompt.
    if next_area in SPECIALISTS and next_area == _last_specialist_area(state["messages"]):
        return {
            "route": "finish",
            "route_reason": (
                f"overridden: {next_area} just answered and the customer has not "
                f"replied (model said: {decision.reason})"
            ),
            "hops": hops + 1,
        }

    update: dict = {
        "route": next_area,
        "route_reason": decision.reason,
        "hops": hops + 1,
    }

    # Handing off to a specialist when the conversation currently ends with an
    # assistant message (billing just spoke, now escalation takes over) would ask
    # the model to continue its own turn. Anthropic rejects that outright as an
    # assistant prefill, and it's incoherent for any provider. So the supervisor
    # states the task explicitly, which is both a valid trailing user turn and a
    # better brief than "here's a transcript, figure it out".
    if next_area in SPECIALISTS and not isinstance(state["messages"][-1], HumanMessage):
        update["messages"] = [
            HumanMessage(
                content=(
                    f"[internal handoff to {next_area}] "
                    f"{decision.task or 'Continue helping the customer.'}"
                ),
                name=HANDOFF_SENDER,
            )
        ]

    return update


def respond(state: SupportState, runtime: Runtime[SupportContext]) -> dict:
    """Reply directly when no specialist is going to.

    Without this node the graph can end a turn in silence: the router picks
    `finish` for a greeting, an out-of-scope question, or a manipulation attempt,
    no specialist runs, and the customer gets nothing back. That's the single most
    embarrassing agent failure mode, and it only shows up on inputs nobody thinks
    to test.

    If a specialist already produced an answer this turn, this is a no-op.
    """
    messages = state["messages"]
    last = messages[-1] if messages else None
    # `text_of`, not `.content`: a message carrying only a thinking block is not an
    # answer, and treating it as one is exactly the silent turn this node exists to
    # prevent.
    already_answered = (
        isinstance(last, AIMessage) and bool(text_of(last)) and not last.tool_calls
    )
    if already_answered:
        return {}

    # No `coerce_context` here: this node uses no tools and reads the name off
    # state, so it has nothing to scope. The line that used to resolve the context
    # and never look at it was also the only thing in the graph that would
    # `AttributeError` on a runtime-less call.
    name = state.get("customer_name", "").split(" ")[0]
    try:
        reply = _router_model().invoke(
            [
                SystemMessage(respond_prompt(name)),
                *visible_messages(messages)[-4:],
            ]
        )
    except Exception:  # noqa: BLE001
        # Last line of defence. This node exists so a turn is never silent, so it
        # is the one place that must not depend on the model being reachable.
        return {"messages": [AIMessage(content=UNREACHABLE_REPLY)]}

    return {"messages": [AIMessage(content=text_of(reply) or UNREACHABLE_REPLY)]}


def route_from_supervisor(
    state: SupportState,
) -> Literal["billing_agent", "merch_agent", "escalation_agent", "respond"]:
    return {
        "billing": "billing_agent",
        "merch": "merch_agent",
        "escalation": "escalation_agent",
    }.get(state.get("route", "finish"), "respond")


def run_config(thread_id: str, **extra) -> dict:
    """The config every caller should invoke this graph with.

    Callers assemble `{"configurable": {"thread_id": ...}}` by hand, and the
    recursion limit is the kind of thing that gets set on one of them and forgotten
    on the others. Building it here means the demo, the eval harness, and anything
    added later share one ceiling — and `GRAPH_RECURSION_LIMIT` derives from
    `MAX_ROUTING_HOPS`, so they also share one reason for its value.
    """
    configurable = {"thread_id": thread_id, **extra.pop("configurable", {})}
    return {
        "configurable": configurable,
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        **extra,
    }


def build_graph(*, checkpointer=None, store=None, host_managed_persistence: bool = False):
    """Assemble and compile the support graph.

    A factory rather than a module-level graph because persistence differs by
    host: `langgraph dev` and LangGraph Platform inject their own checkpointer and
    store, and passing our own would conflict with theirs. The CLI demo supplies
    SQLite-backed ones explicitly.

    Args:
        checkpointer: Thread-scoped persistence. Required for human-in-the-loop.
        store: Cross-thread persistence. The cart and the audit log live here.
        host_managed_persistence: Assert that something else — `langgraph dev`,
            LangGraph Platform — will inject both. Suppresses the warning below.

    Compiling without a checkpointer is legitimate (that is what the module-level
    `graph` below does) but it is also the single most common way to break this
    graph, so it warns rather than passing silently. Two of the tools sit behind
    `HumanInTheLoopMiddleware`, and an interrupt with nowhere to persist to does
    not raise — `issue_refund` and `checkout_cart` simply stop pausing, and the
    approval step this whole system is built around quietly does not happen.
    """
    if checkpointer is None and not host_managed_persistence:
        warnings.warn(
            "build_graph() called without a checkpointer. Refund and checkout "
            "approvals will not interrupt, so money-moving tools run unattended. "
            "Pass checkpointer=..., or host_managed_persistence=True if the "
            "LangGraph server is injecting one.",
            UserWarning,
            stacklevel=2,
        )

    builder = (
        StateGraph(SupportState, context_schema=SupportContext)
        .add_node("authenticate", authenticate)
        .add_node("supervisor", supervisor)
        .add_node("billing_agent", build_billing_agent())
        .add_node("merch_agent", build_merch_agent())
        .add_node("escalation_agent", build_escalation_agent())
        .add_node("respond", respond)
        .add_edge(START, "authenticate")
        .add_edge("authenticate", "supervisor")
        .add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            ["billing_agent", "merch_agent", "escalation_agent", "respond"],
        )
        .add_edge("respond", END)
        # Specialists always report back so the supervisor can decide whether the
        # request is actually finished or needs another area (e.g. billing denies
        # a refund, supervisor routes to escalation).
        .add_edge("billing_agent", "supervisor")
        .add_edge("merch_agent", "supervisor")
        .add_edge("escalation_agent", "supervisor")
    )

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store
    return builder.compile(name="chinook_support", **kwargs)


#: Entrypoint for `langgraph.json` / LangGraph Studio. Persistence is injected by
#: the dev server, so no checkpointer or store here — declared explicitly so this
#: is a statement about the host rather than an omission.
#:
#: `.with_config()` carries the recursion limit because Studio is the one caller
#: that cannot go through `run_config()`: the server builds the config itself from
#: the assistant, and every other entrypoint (the demo, the eval harness) gets the
#: ceiling from `run_config`. Without this, Studio silently ran at LangGraph's
#: default of 25 — the exact value `GRAPH_RECURSION_LIMIT` exists to override, and
#: the one host where a runaway loop is being watched by a person who would read
#: it as the agent thinking hard.
graph = build_graph(host_managed_persistence=True).with_config(
    recursion_limit=GRAPH_RECURSION_LIMIT
)
