"""Tenant-isolation and audit middleware.

`context.py` explains the primary control: identity lives in runtime context, so
the model has no parameter it can set to become someone else. That control is
sound, but it depends on every tool being written correctly forever. The middleware
here is the backstop for the day someone adds `get_invoices(customer_id: int)`
because it was convenient.

`CustomerScopeMiddleware` does two things around every tool call:

  before  reject any call whose arguments carry a customer identifier at all
  after   scan the result for another customer's email address

The post-check is deliberately narrow. Scanning for *names* would false-positive
constantly — Chinook has an artist called "King" and a customer called King. Email
addresses are unique, exact, and unambiguous, which makes the check deterministic
and safe to fail closed on. It's a tripwire, not a sanitizer: the real control is
upstream, and this proves the control held.
"""

from __future__ import annotations

import functools
import json
import uuid
from datetime import datetime
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage

from .context import coerce_context
from .db import query

#: Argument names that would let the model choose whose data to read. No tool in
#: this package accepts any of them; a call containing one is either a bug or an
#: injection attempt, and both should fail loudly.
FORBIDDEN_ARGS = {
    "customer_id",
    "customerid",
    "cust_id",
    "account_id",
    "accountid",
    "user_id",
    "userid",
    "customer_email",
    "on_behalf_of",
}


@functools.lru_cache(maxsize=1)
def _email_owners() -> dict[str, int]:
    """Every customer email in the store, lowercased -> CustomerId.

    Cached for the process lifetime; the customer table doesn't change under us
    during a demo, and re-querying on every tool call would be wasteful.
    """
    return {
        r["Email"].lower(): r["CustomerId"]
        for r in query("SELECT CustomerId, Email FROM Customer WHERE Email IS NOT NULL")
    }


def find_foreign_emails(text: str, caller_id: int) -> list[str]:
    """Return any customer email in `text` that doesn't belong to `caller_id`."""
    haystack = text.lower()
    return [
        email
        for email, owner in _email_owners().items()
        if owner != caller_id and email in haystack
    ]


class ScopeViolation(Exception):
    """Raised internally when a tool call breaches tenant isolation."""


class CustomerScopeMiddleware(AgentMiddleware):
    """Blocks cross-customer data access at the tool boundary.

    Violations are recorded on `runtime.store` under `("security", <customer_id>)`
    so they show up in the audit trail, and returned to the model as a tool error
    so it can explain itself to the customer rather than silently stalling.

    Both `wrap_tool_call` and `awrap_tool_call` are implemented. That is not
    optional: LangGraph's dev server and LangGraph Platform invoke graphs
    asynchronously, and a middleware with only the sync hook raises
    NotImplementedError there while working perfectly in a local script. A
    security control that silently isn't running in production is worse than no
    control, so the two paths share `_precheck`/`_postcheck` and differ only in
    how they await the handler.
    """

    name = "CustomerScopeMiddleware"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        blocked = self._precheck(request)
        if blocked is not None:
            return blocked
        return self._postcheck(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        blocked = self._precheck(request)
        if blocked is not None:
            return blocked
        return self._postcheck(request, await handler(request))

    def _precheck(self, request: ToolCallRequest) -> ToolMessage | None:
        """No tool may be told whose data to fetch."""
        tool_name = request.tool_call.get("name", "<unknown>")
        args = request.tool_call.get("args") or {}
        caller_id = coerce_context(getattr(request.runtime, "context", None)).customer_id

        offending = sorted(FORBIDDEN_ARGS & {k.lower() for k in args})
        if not offending:
            return None

        return self._block(
            request,
            caller_id,
            tool_name,
            reason=(
                f"Tool call rejected: `{tool_name}` was called with "
                f"{', '.join(offending)}. Tools in this system are scoped to the "
                "authenticated customer automatically and never accept a customer "
                "identifier. Retry without that argument. Tell the customer you "
                "can only access their own account."
            ),
            detail={"forbidden_args": offending, "args": _safe(args)},
        )

    def _postcheck(self, request: ToolCallRequest, result: Any) -> Any:
        """Did anything belonging to someone else come back?"""
        tool_name = request.tool_call.get("name", "<unknown>")
        caller_id = coerce_context(getattr(request.runtime, "context", None)).customer_id
        if caller_id is None:
            return result

        leaked = find_foreign_emails(_content_of(result), caller_id)
        if not leaked:
            return result

        return self._block(
            request,
            caller_id,
            tool_name,
            reason=(
                f"Tool result from `{tool_name}` was withheld: it contained data "
                "belonging to another customer. This is a bug in the tool, not "
                "something the customer did. Apologise and offer to escalate to a "
                "support representative."
            ),
            detail={"leaked_emails": leaked},
        )

    def _block(
        self,
        request: ToolCallRequest,
        caller_id: int | None,
        tool_name: str,
        *,
        reason: str,
        detail: dict,
    ) -> ToolMessage:
        _record(
            request,
            namespace=("security", str(caller_id)),
            entry={
                "event": "scope_violation",
                "tool": tool_name,
                "caller_id": caller_id,
                "detail": detail,
                "at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return ToolMessage(
            content=reason,
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
            status="error",
        )


class AuditLogMiddleware(AgentMiddleware):
    """Append-only record of every tool the agent ran on a customer's account.

    Write paths (`issue_refund`, `checkout_cart`, `file_escalation`) move money or
    create obligations. "The agent did it" is not an acceptable answer when someone
    asks why an order exists, so every call lands in the store with its arguments,
    outcome, and who authorised it.
    """

    name = "AuditLogMiddleware"

    def __init__(self, area: str) -> None:
        super().__init__()
        self.area = area

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        started = datetime.now()
        status = "ok"
        try:
            result = handler(request)
            if getattr(result, "status", None) == "error":
                status = "error"
            return result
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            status = f"exception:{type(exc).__name__}"
            raise
        finally:
            self._write(request, started, status)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        started = datetime.now()
        status = "ok"
        try:
            result = await handler(request)
            if getattr(result, "status", None) == "error":
                status = "error"
            return result
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            status = f"exception:{type(exc).__name__}"
            raise
        finally:
            self._write(request, started, status)

    def _write(self, request: ToolCallRequest, started: datetime, status: str) -> None:
        context = coerce_context(getattr(request.runtime, "context", None))
        _record(
            request,
            namespace=("audit", str(context.customer_id)),
            entry={
                "area": self.area,
                "tool": request.tool_call.get("name"),
                "args": _safe(request.tool_call.get("args") or {}),
                "status": status,
                "channel": context.channel,
                "acting_staff": context.staff_agent_email,
                "at": started.isoformat(timespec="seconds"),
            },
        )


# --- helpers -----------------------------------------------------------------


def _record(request: ToolCallRequest, *, namespace: tuple, entry: dict) -> None:
    """Best-effort write to the runtime store.

    Audit logging must never be the reason a customer's request fails, so this
    swallows store errors. In production this would emit to a real append-only
    sink and alert on write failure instead.
    """
    store = getattr(request.runtime, "store", None)
    if store is None:
        return
    try:
        store.put(namespace, uuid.uuid4().hex, entry)
    except Exception:  # noqa: BLE001
        pass


def _content_of(result: Any) -> str:
    """Flatten whatever a tool returned into text we can scan."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except Exception:  # noqa: BLE001
        return str(content)


def _safe(args: dict) -> dict:
    """Truncate tool args so a huge payload can't bloat the audit record."""
    out = {}
    for key, value in args.items():
        text = str(value)
        out[key] = text if len(text) <= 200 else text[:200] + "..."
    return out
