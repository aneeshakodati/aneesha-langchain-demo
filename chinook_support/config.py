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
#: Supersteps LangGraph will run before raising `GraphRecursionError`.
#:
#: The hop limit in `supervisor` already bounds this graph, so under correct
#: behaviour the ceiling is never reached: authenticate (1) + MAX_ROUTING_HOPS x
#: (supervisor + specialist) (8) + the supervisor turn that says "finish" (1) +
#: respond (1) = 11. The point is that it holds when the hop limit *doesn't* --
#: a reducer that stops resetting `hops`, an edge added back to a specialist. The
#: default is 25, which is neither derived from this graph nor obviously wrong, so
#: it would absorb that bug as a slow, expensive turn instead of an error.
#:
#: Set from MAX_ROUTING_HOPS so raising one raises the other; the +4 is headroom for
#: nodes added around the loop, not for extra trips through it.
GRAPH_RECURSION_LIMIT = 2 * MAX_ROUTING_HOPS + 7
