"""What happens to a turn when the model provider is unreachable.

`make demo` died mid-run on a single transient `AnthropicConnectionError` raised
inside the supervisor. The specialists were covered — `ToolRetryMiddleware` — but
`supervisor` and `respond` call the model directly and had nothing around them, so
one network blip took the whole turn down. In front of an audience.

These tests simulate that by making the model raise, which is the only way to
exercise it: you cannot ask the real API to fail on cue, so the path that runs
during an outage is the path that never gets tested.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from chinook_support.graph import UNREACHABLE_REPLY, respond, supervisor, text_of


class Boom(Exception):
    """Stands in for a transport-level provider error."""


def _always_fails(*_args, **_kwargs):
    raise Boom("connection error")


AUTHED = {
    "messages": [HumanMessage("what have I bought recently?")],
    "authenticated": True,
    "hops": 0,
    "customer_name": "Luís Gonçalves",
}


def test_the_router_falls_back_instead_of_raising():
    """A dead router must not take the conversation with it.

    Routing is the one step with a safe default: send the turn to `respond` and let
    the customer be told something. The alternative is a stack trace where a reply
    should be.
    """
    with patch("chinook_support.graph._router_model", side_effect=Boom("down")):
        update = supervisor(AUTHED, runtime=None)

    assert update["route"] == "finish"
    assert "router unavailable" in update["route_reason"]
    assert "Boom" in update["route_reason"]


def test_the_router_burns_its_hop_on_the_way_down():
    """Otherwise a provider having a bad minute is an infinite loop.

    `finish` routes to `respond`, which ends the turn, so this is belt and braces —
    but the hop counter is the only thing standing between a persistent fault and a
    loop, and it costs one line.
    """
    with patch("chinook_support.graph._router_model", side_effect=Boom("down")):
        update = supervisor({**AUTHED, "hops": 2}, runtime=None)

    assert update["hops"] == 3


def test_the_customer_still_gets_a_sentence():
    """`respond` exists so a turn is never silent.

    That makes it the one node that must not depend on the model being reachable —
    a fallback that itself needs the thing that just failed is not a fallback.
    """
    with patch("chinook_support.graph._router_model", side_effect=Boom("down")):
        update = respond(AUTHED, runtime=None)

    (message,) = update["messages"]
    assert text_of(message) == UNREACHABLE_REPLY


def test_an_empty_reply_is_treated_as_no_reply():
    """A model can return successfully and say nothing.

    Rare, but the result is the silent turn this node exists to prevent, so it gets
    the same treatment as an outage.
    """

    class Empty:
        content = ""
        text = ""

    with patch("chinook_support.graph._router_model", return_value=_Returns(Empty())):
        update = respond(AUTHED, runtime=None)

    (message,) = update["messages"]
    assert text_of(message) == UNREACHABLE_REPLY


def test_a_specialist_reply_short_circuits_respond():
    """No second reply when a specialist already answered."""
    answered = {**AUTHED, "messages": [*AUTHED["messages"], AIMessage("Here you go.")]}

    with patch("chinook_support.graph._router_model", side_effect=_always_fails):
        assert respond(answered, runtime=None) == {}


class _Returns:
    """Minimal stand-in for the retry-wrapped runnable."""

    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value
