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

#: Specialist agents. Reasoning quality matters most here.
AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic:claude-sonnet-5")


def agent_model() -> str:
    """The specialist model, resolved at agent-construction time.

    Read through a function rather than off the module constant so that
    `run_eval.py --model ...` works: the constant is frozen at import, and by the
    time the flag is parsed `agents.py` has long since imported it. That silently
    made the Sonnet-vs-Haiku comparison — the whole point of the flag — run both
    experiments on the same model.
    """
    return os.getenv("AGENT_MODEL", AGENT_MODEL)
#: Supervisor router. One structured-output call per hop, so latency/cost dominate.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "anthropic:claude-haiku-4-5-20251001")
#: Conversation summarization once the merch browse session gets long.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "anthropic:claude-haiku-4-5-20251001")
#: LLM-as-judge in the eval suite.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic:claude-sonnet-5")

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
