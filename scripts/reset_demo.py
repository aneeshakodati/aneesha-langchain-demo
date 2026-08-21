"""Reset demo state so the whole script can be run again cleanly.

Rebuilds `chinook_demo.db` from the pristine copy (dropping refunds, support cases,
and any orders placed during the last run) and clears the checkpointer and store
databases, which is what wipes saved carts and conversation history.

Run this between rehearsals and immediately before presenting.

    python scripts/reset_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chinook_support.config import CHECKPOINT_DB, DEMO_DB, PRISTINE_DB, STORE_DB  # noqa: E402


def main() -> None:
    if not PRISTINE_DB.exists():
        print("No pristine database yet. Run: python scripts/build_db.py")
        raise SystemExit(1)

    from scripts.build_db import build_demo, verify

    print("Resetting demo state")
    build_demo()

    for path in (CHECKPOINT_DB, STORE_DB):
        if path.exists():
            path.unlink()
            print(f"  cleared {path.name}")
        else:
            print(f"  {path.name} already absent")

    verify()
    print(f"Done. {DEMO_DB.name} rebuilt; carts and conversations cleared.")


if __name__ == "__main__":
    main()
