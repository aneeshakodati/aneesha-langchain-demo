"""Database access.

Three rules, enforced structurally rather than by convention:

1. Reads go through a connection opened in SQLite read-only mode. A bug in a
   "read" tool cannot mutate anything, because the file handle won't allow it.
2. Every query is parameterized. There is no string interpolation of values
   anywhere in this package, and no tool exposes SQL to the model.
3. *Which* database is in play is a per-context decision, not a global one. See
   `use_db` below.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from .config import DEMO_DB

#: The database this context is bound to. `None` means "the shared demo database",
#: which is what the CLI demo and Studio always want.
#:
#: The eval suite wants something else. Its examples mutate the store — refunds,
#: support cases, orders — so running two of them against one file makes each
#: example's writes visible to the other, and the suite grades correct behaviour
#: as a failure at random. The previous fix was a per-customer mutex, which worked
#: but serialized eleven of the thirty-five cases behind one lock. Giving each
#: example a private copy of the database removes the shared resource instead of
#: queueing for it, and the suite runs at full concurrency.
#:
#: A ContextVar rather than a parameter threaded through forty call sites: the
#: binding has to reach `policy.adjudicate` and `cart.build_cart` too, and those
#: are pure functions that should not grow a `db=` argument to serve a test
#: harness.
_ACTIVE_DB: ContextVar[Path | None] = ContextVar("chinook_active_db", default=None)


def active_db() -> Path:
    """The database path this context should read and write."""
    return _ACTIVE_DB.get() or DEMO_DB


@contextmanager
def use_db(path: str | Path | None) -> Iterator[Path]:
    """Bind this context to `path`. A `None` path is a no-op, so callers that may
    or may not have a private database can wrap unconditionally.

    ContextVars are per-thread, so a worker in an evaluation thread pool binds
    only itself. That is exactly the isolation the eval suite needs.
    """
    if path is None:
        yield active_db()
        return
    resolved = Path(path)
    token = _ACTIVE_DB.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_DB.reset(token)


def _row_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@contextmanager
def read_conn() -> Iterator[sqlite3.Connection]:
    """Open the active database read-only.

    Uses a URI with `mode=ro` so SQLite itself rejects writes. This is the
    connection every lookup tool uses.
    """
    conn = sqlite3.connect(f"file:{active_db()}?mode=ro", uri=True)
    conn.row_factory = _row_factory
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_conn() -> Iterator[sqlite3.Connection]:
    """Open the active database read-write, in a transaction.

    Only three tools ever reach this: `issue_refund`, `checkout_cart`, and
    `file_escalation` — all of which sit behind human approval or write to
    support tables rather than to customer-facing financial records.
    """
    conn = sqlite3.connect(active_db())
    conn.row_factory = _row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Run a parameterized read query and return rows as dicts."""
    with read_conn() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    with read_conn() as conn:
        return conn.execute(sql, params).fetchone()


def money(value: Any) -> Decimal:
    """Chinook stores currency as NUMERIC, which sqlite3 hands back as float.

    Round-trip through `str` so we never do arithmetic on binary floats — a
    refund that's off by a cent is a support ticket of its own.
    """
    return Decimal(str(value or 0)).quantize(Decimal("0.01"))


# --- Customer lookups --------------------------------------------------------


def get_customer(customer_id: int) -> dict[str, Any] | None:
    """Load the authenticated customer's own profile row."""
    return query_one(
        """
        SELECT c.CustomerId, c.FirstName, c.LastName, c.Email, c.Country,
               c.City, c.State, c.Phone, c.SupportRepId,
               e.FirstName AS RepFirstName, e.LastName AS RepLastName,
               e.Email AS RepEmail, e.Title AS RepTitle
          FROM Customer c
          LEFT JOIN Employee e ON e.EmployeeId = c.SupportRepId
         WHERE c.CustomerId = ?
        """,
        (customer_id,),
    )


def owns_invoice(customer_id: int, invoice_id: int) -> bool:
    """Authorization check: does this invoice belong to this customer?

    Called by every tool that takes an `invoice_id`. An invoice id is the one
    identifier a customer legitimately knows and types, so it *is* a tool
    parameter — which means it has to be checked on every single use.
    """
    row = query_one(
        "SELECT 1 AS ok FROM Invoice WHERE InvoiceId = ? AND CustomerId = ?",
        (invoice_id, customer_id),
    )
    return row is not None


def owned_track_ids(customer_id: int) -> set[int]:
    """Every track this customer has already purchased.

    Used by the cart builder to honor "nothing I already own".
    """
    rows = query(
        """
        SELECT DISTINCT il.TrackId
          FROM InvoiceLine il
          JOIN Invoice i ON i.InvoiceId = il.InvoiceId
         WHERE i.CustomerId = ?
        """,
        (customer_id,),
    )
    return {r["TrackId"] for r in rows}
