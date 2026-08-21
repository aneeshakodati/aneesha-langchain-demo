"""Central configuration: paths, model ids, and business policy thresholds.

Policy thresholds live here rather than in a prompt on purpose. A number the LLM
reads out of a system prompt is a suggestion; a number Python compares against is
a rule. See `policy.py`.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

# --- Paths -------------------------------------------------------------------

CHINOOK_SQL = DATA_DIR / "Chinook_Sqlite.sql"
#: Pristine Chinook, never written to. Rebuilt only by scripts/build_db.py.
PRISTINE_DB = DATA_DIR / "chinook.db"
#: Working copy the agent reads and writes. Reset by scripts/reset_demo.py.
DEMO_DB = DATA_DIR / "chinook_demo.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"
STORE_DB = DATA_DIR / "store.sqlite"

# --- Models ------------------------------------------------------------------
#
# Every model id is read through an accessor, never off a module constant.
#
# The constant form is a trap, and this repo already fell into it once: `--model`
# on `run_eval.py` sets `AGENT_MODEL` in the environment long after `agents.py`
# has imported it, so the flag silently did nothing and the Sonnet-vs-Haiku
# comparison — the flag's whole purpose — ran both experiments on Sonnet. The fix
# is applied to all four, not just the one that had a caller, because the next
# override flag shouldn't have to rediscover it.
#
# The `*_MODEL` names below are the *defaults*. Read them with the functions.

#: Specialist agents. Reasoning quality matters most here.
AGENT_MODEL = "anthropic:claude-sonnet-5"
#: Supervisor router. One structured-output call per hop, so latency/cost dominate.
ROUTER_MODEL = "anthropic:claude-haiku-4-5-20251001"
#: Conversation summarization once the merch browse session gets long.
SUMMARY_MODEL = "anthropic:claude-haiku-4-5-20251001"
#: LLM-as-judge in the eval suite.
JUDGE_MODEL = "anthropic:claude-sonnet-5"


def agent_model() -> str:
    """The specialist model, resolved at agent-construction time."""
    return os.getenv("AGENT_MODEL", AGENT_MODEL)


def router_model() -> str:
    """The supervisor/respond model, resolved per call."""
    return os.getenv("ROUTER_MODEL", ROUTER_MODEL)


def summary_model() -> str:
    """The summarization model, resolved at middleware-construction time."""
    return os.getenv("SUMMARY_MODEL", SUMMARY_MODEL)


def judge_model() -> str:
    """The eval judge model, resolved per evaluation."""
    return os.getenv("JUDGE_MODEL", JUDGE_MODEL)

# --- Refund policy -----------------------------------------------------------

#: Orders older than this are never auto-refundable; they route to a human.
REFUND_WINDOW_DAYS = 30
#: At or below this amount, an eligible refund needs no human approval.
REFUND_AUTO_APPROVE_LIMIT = Decimal("10.00")
#: Refuse outright above this — no agent, approved or not, moves this much money.
REFUND_HARD_CEILING = Decimal("100.00")

# --- Cart policy -------------------------------------------------------------

MAX_CART_ITEMS = 40
#: Any checkout requires human approval regardless of amount. It creates a real order.
#:
#: Read by `merch_middleware()`, which is the point: this was a module constant
#: nothing imported, so it read like the switch that controlled checkout approval
#: while the actual gate was hard-coded in the middleware. Editing it did nothing.
#:
#: It is a real switch now, so treat it like one. Setting it False removes the only
#: thing between the agent and a real order — there is no amount ceiling on checkout
#: the way there is on refunds, because the human *was* the ceiling.
CHECKOUT_ALWAYS_REQUIRES_APPROVAL = True

# --- Agent safety rails ------------------------------------------------------

#: Ceiling on supervisor -> specialist -> supervisor round trips per user turn.
MAX_ROUTING_HOPS = 4
#: Model calls per specialist, per run. Runaway-loop and cost protection.
MAX_MODEL_CALLS_PER_RUN = 8
#: Escalation is a short, fixed routine — look up the rep, file the ticket, tell the
#: customer — so it gets a tighter ceiling than the open-ended areas. It was 4, which
#: left no room for a tool retry: the eval suite caught a run that filed the ticket
#: and then ran out of budget before it could say so, ending the turn on a raw
#: "Model call limits exceeded" string. Six covers the routine plus one retry.
MAX_MODEL_CALLS_ESCALATION = 6
#: Catalog searches per run, so browsing can't grind.
MAX_CATALOG_SEARCHES_PER_RUN = 6
#: Nodes in the longest specialist ReAct cycle. Escalation's is the deepest: the
#: two PII middlewares contribute a `before_model` and an `after_model` node each,
#: the call budget another of both, plus `model` and `tools`. Every one of them is
#: a superstep, and the cycle runs once per model call.
#:
#: Pinned against the compiled agents by `tests/test_middleware.py` rather than
#: counted by hand here, because the number changes whenever a middleware is added
#: and the cost of it being stale is the incident described below.
LONGEST_SPECIALIST_CYCLE = 8
#: Supersteps LangGraph will run before raising `GraphRecursionError`.
#:
#: One value has to cover every graph in the system, specialists included:
#: `recursion_limit` propagates down into subgraphs, each subgraph counts its own
#: supersteps against it, and a subgraph cannot raise it back — `.with_config()` on
#: a graph attached with `add_node` loses to the config coming down from the parent.
#:
#: This used to be `2 * MAX_ROUTING_HOPS + 7`, and the arithmetic behind it was
#: right about the parent: authenticate (1) + MAX_ROUTING_HOPS x (supervisor +
#: specialist) (8) + the supervisor turn that says "finish" (1) + respond (1) = 11,
#: so 15 left headroom. It was measuring the wrong graph. A specialist is a
#: `create_agent` loop that spends a whole cycle per model call, so escalation's six
#: calls need ~51 supersteps on their own. Experiment `chinook-full-50137e33` failed
#: 11 of 35 cases on `GraphRecursionError` in `escalation_agent`: route accuracy
#: 1.00 -> 0.70, and the escalation judge to 0.00, because the subgraph died before
#: `file_escalation` ran and the tickets the judge grades were never filed.
#:
#: So: longest cycle x largest call budget, plus one cycle for the final trip
#: through `before_model` that finds the budget spent and exits. That is loose for
#: the parent graph on purpose. `MAX_ROUTING_HOPS` bounds the routing loop and
#: `CallBudgetMiddleware` bounds each specialist; this is the backstop for when one
#: of *those* breaks, which is the job the old comment claimed and the old number
#: could not do.
GRAPH_RECURSION_LIMIT = LONGEST_SPECIALIST_CYCLE * (
    max(MAX_MODEL_CALLS_PER_RUN, MAX_MODEL_CALLS_ESCALATION) + 1
)
