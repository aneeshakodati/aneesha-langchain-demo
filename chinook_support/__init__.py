"""Chinook Records customer support agent.

A LangChain/LangGraph proof of concept: a multi-area support bot over the Chinook
music-store dataset, built to demonstrate reliable agent engineering rather than
breadth of tooling. See README.md for the design rationale.
"""

import warnings

# LangGraph serializes the runtime context when checkpointing a subgraph, and
# `create_agent`'s internal state model annotates that field loosely enough that
# Pydantic emits a serializer warning for any non-None context. It is cosmetic —
# the context arrives at nodes and tools correctly, which `tests/test_scoping.py`
# asserts — but it prints on every single turn, which is unusable in a live demo.
#
# The filter anchors on the message prefix: `filterwarnings` matches with
# `re.match`, and the warning text is multi-line, so a leading `.*` would not
# reach the interesting part.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings",
    category=UserWarning,
)

__all__ = ["build_graph"]


def __getattr__(name: str):
    if name == "build_graph":
        from .graph import build_graph

        return build_graph
    raise AttributeError(name)
