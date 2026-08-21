"""The graph object `langgraph.json` hands to the dev server and Studio.

Every other caller reaches the graph through `run_config()`, which is where the
recursion ceiling, the thread id, and anything added later get set. Studio cannot:
the LangGraph server builds the run config itself from the assistant, so whatever
`run_config` sets is simply absent there.

That makes the Studio export the one entrypoint whose configuration has to travel
*on the graph object*, and the one that regresses silently — a missing ceiling is
not an error, it is LangGraph's default of 25, which is exactly the value
`GRAPH_RECURSION_LIMIT` was written to override. A runaway routing loop under that
default is a slow expensive turn in front of whoever is watching the demo, not a
`GraphRecursionError`.

Verified against a running `langgraph dev`: with the limit bound the server raises
at the bound value; without it, at 25.
"""

from __future__ import annotations

import json
from pathlib import Path

from chinook_support.config import GRAPH_RECURSION_LIMIT
from chinook_support.graph import graph

ROOT = Path(__file__).resolve().parent.parent


def test_langgraph_json_points_at_the_exported_graph():
    """A rename here breaks Studio and nothing else, so nothing else would catch it."""
    spec = json.loads((ROOT / "langgraph.json").read_text())
    assert spec["graphs"]["support"] == "chinook_support.graph:graph"


def test_the_studio_graph_carries_the_recursion_ceiling():
    """Studio never calls `run_config()`, so the ceiling has to ride on the graph."""
    assert graph.config.get("recursion_limit") == GRAPH_RECURSION_LIMIT


def test_the_ceiling_is_not_langgraphs_default():
    """Guards the assertion above from being satisfied by an accident.

    If `GRAPH_RECURSION_LIMIT` ever drifted to 25, the test above would still pass
    while the binding had stopped meaning anything.
    """
    assert GRAPH_RECURSION_LIMIT != 25


def test_binding_config_left_the_graph_usable():
    """`.with_config()` returns a wrapper, not the graph.

    The dev server drives this object for state reads, interrupts and resumes, so
    a wrapper that dropped any of that would break human-in-the-loop in Studio
    while every in-process test kept passing.
    """
    for attribute in ("invoke", "ainvoke", "stream", "astream", "get_state", "update_state"):
        assert hasattr(graph, attribute), attribute
    assert graph.name == "chinook_support"
