"""The three specialists, each built with `create_agent()`.

Splitting the bot into specialists is not decoration. Two reasons:

1. Tool-selection accuracy. Each agent below sees four to six tools. A single flat
   agent would see twelve, and tool confusion is the most common way agents fail
   in production — not hallucination, just picking the wrong function.
2. Policy differs by area. Compare the middleware stacks in `middleware.py`:
   billing has human approval and an audit trail, merch has a search budget and
   summarization, escalation has PII redaction. Those are real differences, not
   settings that could be merged.
"""

from __future__ import annotations

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from .config import AGENT_MODEL
from .context import SupportContext
from .middleware import (
    billing_middleware,
    escalation_middleware,
    merch_middleware,
)
from .tools.billing import BILLING_TOOLS
from .tools.escalation import ESCALATION_TOOLS
from .tools.merch import MERCH_TOOLS


class RouteDecision(BaseModel):
    """Structured output for the supervisor.

    Structured output rather than a tool call: the router does exactly one thing,
    so there is nothing to choose between, and a schema-constrained response is
    cheaper, faster, and — most usefully — turns routing into a deterministic
    eval metric (`evals/evaluators.py::route_accuracy`).
    """

    next: Literal["billing", "merch", "escalation", "finish"] = Field(
        description="Which specialist should act next, or finish if we are done."
    )
    reason: str = Field(
        description="One short sentence explaining the choice, for the trace."
    )
    task: str = Field(
        default="",
        description=(
            "One sentence telling the chosen specialist what to do, phrased as an "
            "instruction. Empty when finishing."
        ),
    )


def build_billing_agent():
    """Orders, charges, refund adjudication."""
    return create_agent(
        model=AGENT_MODEL,
        tools=BILLING_TOOLS,
        middleware=billing_middleware(),
        context_schema=SupportContext,
        name="billing_agent",
    )


def build_merch_agent():
    """Catalog search and constraint-based cart building."""
    return create_agent(
        model=AGENT_MODEL,
        tools=MERCH_TOOLS,
        middleware=merch_middleware(),
        context_schema=SupportContext,
        name="merch_agent",
    )


def build_escalation_agent():
    """Handoff to the customer's assigned human representative.

    The ticket's shape is enforced by `file_escalation`'s own signature — Literal
    enums for category/severity/sentiment plus required narrative fields — so the
    schema is validated at the tool boundary and the structured record is visible
    both in the trace and in the SupportCase table. That leaves the agent's final
    message free to be ordinary prose addressed to the customer, which is what the
    parent graph needs to return.
    """
    return create_agent(
        model=AGENT_MODEL,
        tools=ESCALATION_TOOLS,
        middleware=escalation_middleware(),
        context_schema=SupportContext,
        name="escalation_agent",
    )
