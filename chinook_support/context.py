"""Runtime context — who the agent is acting on behalf of.

This is the single most important file for the security story.

`customer_id` arrives here from the *application's* authenticated session and is
passed to the graph as runtime context, not as state and not as a tool argument.
The model never sees a parameter it could set to change identity: there is simply
no token it can emit that makes `runtime.context.customer_id` something else.

Contrast with the common (broken) pattern of `get_invoices(customer_id: int)`,
where the model decides whose data to fetch and the only thing standing between a
customer and someone else's purchase history is the prompt.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SupportContext(BaseModel):
    """Immutable, per-request identity. Set by the caller, never by the model.

    A Pydantic model rather than a dataclass for two practical reasons: LangGraph
    Studio renders it as a form from the JSON schema (so you can switch customer
    mid-demo), and it coerces the plain dict the Studio server sends into a typed
    object before any node sees it.

    Attributes:
        customer_id: Chinook `Customer.CustomerId` of the authenticated caller.
            `None` means unauthenticated — the graph refuses to do any data work.
        channel: Where the conversation came from. Only used for the audit trail.
        staff_agent_email: Set when a human support rep is driving on a customer's
            behalf. Recorded in the audit log so impersonation is attributable.
            It does NOT widen data access.
    """

    customer_id: Optional[int] = Field(
        default=None,
        description="Authenticated Chinook CustomerId. Try 1 (Luís, Brazil) or 2 (Leonie, Germany).",
    )
    channel: Literal["web", "email", "phone", "studio"] = Field(
        default="web", description="Where the conversation originated."
    )
    staff_agent_email: Optional[str] = Field(
        default=None,
        description="Set only when a human rep is driving. Does not widen data access.",
    )


def require_customer_id(context: SupportContext | dict | None) -> int:
    """Extract the authenticated customer id or raise.

    Tools call this instead of reading `context.customer_id` directly so that an
    unauthenticated request fails loudly at the boundary rather than quietly
    running a query with `customer_id = None`.

    LangGraph hands context through as either the dataclass or a plain dict
    depending on how the run was started (Studio sends JSON), so handle both.
    """
    if context is None:
        raise PermissionError("No runtime context: cannot determine the caller.")

    customer_id = (
        context.get("customer_id")
        if isinstance(context, dict)
        else getattr(context, "customer_id", None)
    )

    if customer_id is None:
        raise PermissionError(
            "Unauthenticated request: no customer_id in runtime context."
        )
    return int(customer_id)


def coerce_context(context: SupportContext | dict | None) -> SupportContext:
    """Normalize whatever LangGraph handed us into a `SupportContext`.

    LangGraph coerces the context into the declared schema before nodes run, but
    middleware can be invoked in situations where it's still a dict or absent
    (outside a graph, in tests), so normalize defensively rather than assuming.
    """
    if isinstance(context, SupportContext):
        return context
    if isinstance(context, dict):
        return SupportContext.model_validate(context)
    return SupportContext()
