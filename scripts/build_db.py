"""Build the Chinook databases.

    data/Chinook_Sqlite.sql  ->  data/chinook.db        (pristine, never written)
                             ->  data/chinook_demo.db   (working copy + support tables)

Chinook's newest invoice is dated 2025-12-22, so relative to any present-day run
*every* historical order falls outside a 30-day refund window. Rather than shifting
all the dates (which would misrepresent the dataset), we seed a handful of recent
orders so that all three refund branches — auto-approve, needs-approval, and
out-of-window — are reachable deterministically.

Run:  python scripts/build_db.py
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chinook_support.config import CHINOOK_SQL, DEMO_DB, PRISTINE_DB  # noqa: E402

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

#: Support tables Chinook doesn't ship. These are ours; the agent writes here.
SUPPORT_SCHEMA = """
CREATE TABLE IF NOT EXISTS Refund (
    RefundId      INTEGER PRIMARY KEY AUTOINCREMENT,
    InvoiceId     INTEGER NOT NULL REFERENCES Invoice(InvoiceId),
    CustomerId    INTEGER NOT NULL REFERENCES Customer(CustomerId),
    Amount        NUMERIC(10,2) NOT NULL,
    Reason        TEXT NOT NULL,
    ApprovedBy    TEXT NOT NULL,          -- 'policy:auto' or 'human:<who>'
    CreatedAt     DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS SupportCase (
    CaseId        INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerId    INTEGER NOT NULL REFERENCES Customer(CustomerId),
    AssignedRepId INTEGER REFERENCES Employee(EmployeeId),
    Category      TEXT NOT NULL,
    Severity      TEXT NOT NULL,
    Sentiment     TEXT NOT NULL,
    Subject       TEXT NOT NULL,
    Summary       TEXT NOT NULL,
    StepsTaken    TEXT NOT NULL,
    Recommendation TEXT NOT NULL,
    RelatedInvoices TEXT,
    Status        TEXT NOT NULL DEFAULT 'open',
    CreatedAt     DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_refund_customer ON Refund(CustomerId);
CREATE INDEX IF NOT EXISTS ix_case_customer ON SupportCase(CustomerId);
CREATE INDEX IF NOT EXISTS ix_invoiceline_track ON InvoiceLine(TrackId);
CREATE INDEX IF NOT EXISTS ix_invoice_customer_date ON Invoice(CustomerId, InvoiceDate);
"""

#: (customer_id, days_ago, n_tracks, starting_track_id) -> which refund branch it exercises
SEED_ORDERS = [
    (1, 5, 6, 3000),  # ~$5.94  -> eligible, under the auto-approve limit
    (1, 12, 26, 2500),  # ~$25.74 -> eligible, over the limit, needs a human
    (2, 3, 14, 1200),  # ~$13.86 -> eligible, over the limit, needs a human
    (3, 200, 8, 900),  # outside the 30-day window -> must escalate
]


def download_sql() -> None:
    if CHINOOK_SQL.exists() and CHINOOK_SQL.stat().st_size > 100_000:
        print(f"  have {CHINOOK_SQL.name}")
        return
    CHINOOK_SQL.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {CHINOOK_URL}")
    subprocess.run(["curl", "-sSL", "-o", str(CHINOOK_SQL), CHINOOK_URL], check=True)


def build_pristine() -> None:
    if PRISTINE_DB.exists():
        PRISTINE_DB.unlink()
    print(f"  building {PRISTINE_DB.name}")
    sql = CHINOOK_SQL.read_text(encoding="utf-8", errors="replace")
    conn = sqlite3.connect(PRISTINE_DB)
    conn.executescript(sql)
    conn.commit()
    conn.close()


def build_demo() -> None:
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    print(f"  building {DEMO_DB.name}")
    src = sqlite3.connect(PRISTINE_DB)
    dst = sqlite3.connect(DEMO_DB)
    src.backup(dst)
    src.close()

    dst.executescript(SUPPORT_SCHEMA)
    seed_recent_orders(dst)
    dst.commit()
    dst.close()


def seed_recent_orders(conn: sqlite3.Connection) -> None:
    """Add present-day orders so refund policy branches are all reachable."""
    now = datetime.now()
    for customer_id, days_ago, n_tracks, first_track in SEED_ORDERS:
        cust = conn.execute(
            "SELECT Address, City, State, Country, PostalCode "
            "FROM Customer WHERE CustomerId = ?",
            (customer_id,),
        ).fetchone()
        if cust is None:
            continue
        address, city, state, country, postal = cust

        tracks = conn.execute(
            "SELECT TrackId, UnitPrice FROM Track WHERE TrackId >= ? "
            "ORDER BY TrackId LIMIT ?",
            (first_track, n_tracks),
        ).fetchall()
        total = round(sum(price for _, price in tracks), 2)
        invoice_date = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")

        cur = conn.execute(
            "INSERT INTO Invoice (CustomerId, InvoiceDate, BillingAddress, "
            "BillingCity, BillingState, BillingCountry, BillingPostalCode, Total) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (customer_id, invoice_date, address, city, state, country, postal, total),
        )
        invoice_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity) "
            "VALUES (?, ?, ?, 1)",
            [(invoice_id, tid, price) for tid, price in tracks],
        )
        print(
            f"    seeded invoice {invoice_id}: customer {customer_id}, "
            f"{days_ago}d ago, ${total}"
        )


def verify() -> None:
    conn = sqlite3.connect(DEMO_DB)
    expected = {"Customer": 59, "Track": 3503, "Employee": 8}
    for table, count in expected.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert actual == count, f"{table}: expected {count}, got {actual}"
        print(f"    {table}: {actual}")
    invoices = conn.execute("SELECT COUNT(*) FROM Invoice").fetchone()[0]
    assert invoices == 412 + len(SEED_ORDERS), f"Invoice: got {invoices}"
    print(f"    Invoice: {invoices} (412 original + {len(SEED_ORDERS)} seeded)")
    for table in ("Refund", "SupportCase"):
        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"    {table}: present")
    conn.close()


def main() -> None:
    print("Chinook database build")
    download_sql()
    build_pristine()
    build_demo()
    print("  verifying")
    verify()
    print("Done.")


if __name__ == "__main__":
    main()
