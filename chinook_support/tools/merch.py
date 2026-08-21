"""Merchandising tools — catalog search, constraint-based cart building, checkout.

The interesting tool here is `build_music_cart`. The model's job is to turn "some
jazz and blues, keep it under fifteen bucks, nothing I've already got" into a
`CartConstraints`; the solver in `cart.py` does the arithmetic. The model never
adds up prices, because models cannot reliably add up 12 prices.

The cart lives in the LangGraph Store under `("cart", <customer_id>)` rather than
in graph state. That's deliberate: a cart should outlive the conversation. Come
back tomorrow on a brand new thread and it's still there, which is both realistic
and a neat demonstration of cross-thread persistence.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation

from langchain.tools import ToolRuntime, tool

from ..cart import CartConstraints, build_cart
from ..config import MAX_CART_ITEMS
from ..context import require_customer_id
from ..db import money, query, write_conn

CART_KEY = "current"


def _cart_ns(customer_id: int) -> tuple[str, str]:
    return ("cart", str(customer_id))


def _load_cart(runtime: ToolRuntime, customer_id: int) -> list[int]:
    if runtime.store is None:
        return []
    item = runtime.store.get(_cart_ns(customer_id), CART_KEY)
    return list(item.value.get("track_ids", [])) if item else []


def _save_cart(runtime: ToolRuntime, customer_id: int, track_ids: list[int]) -> None:
    if runtime.store is None:
        return
    runtime.store.put(
        _cart_ns(customer_id),
        CART_KEY,
        {
            "track_ids": track_ids,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _hydrate(track_ids: list[int]) -> list[dict]:
    """Turn track ids into displayable rows, preserving cart order."""
    if not track_ids:
        return []
    placeholders = ",".join("?" * len(track_ids))
    rows = query(
        f"""
        SELECT t.TrackId, t.Name AS Title, t.UnitPrice,
               ar.Name AS Artist, al.Title AS Album,
               COALESCE(g.Name,'Unknown') AS Genre
          FROM Track t
          JOIN Album al  ON al.AlbumId  = t.AlbumId
          JOIN Artist ar ON ar.ArtistId = al.ArtistId
          LEFT JOIN Genre g ON g.GenreId = t.GenreId
         WHERE t.TrackId IN ({placeholders})
        """,
        tuple(track_ids),
    )
    by_id = {r["TrackId"]: r for r in rows}
    return [
        {
            "track_id": tid,
            "title": by_id[tid]["Title"],
            "artist": by_id[tid]["Artist"],
            "album": by_id[tid]["Album"],
            "genre": by_id[tid]["Genre"],
            "price": str(money(by_id[tid]["UnitPrice"])),
        }
        for tid in track_ids
        if tid in by_id
    ]


def _cart_summary(track_ids: list[int]) -> dict:
    items = _hydrate(track_ids)
    total = sum((Decimal(i["price"]) for i in items), Decimal("0.00"))
    return {
        "items": items,
        "track_count": len(items),
        "total": str(total.quantize(Decimal("0.01"))),
    }


def _to_decimal(value) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


@tool
def search_catalog(
    runtime: ToolRuntime,
    search: str = "",
    genre: str = "",
    artist: str = "",
    limit: int = 10,
) -> dict:
    """Search the store catalog by track title, genre, and/or artist.

    Use this for open-ended browsing ("do you have any Miles Davis?"). To assemble
    a set of tracks under a budget, use `build_music_cart` instead — it handles the
    constraints properly.

    Args:
        search: Words to match against the track title.
        genre: Genre name, e.g. "Jazz". Partial matches work.
        artist: Artist name. Partial matches work.
        limit: Maximum results (1-25).
    """
    require_customer_id(runtime.context)
    limit = max(1, min(int(limit), 25))

    where = ["1=1"]
    params: list = []
    if search:
        where.append("t.Name LIKE ?")
        params.append(f"%{search}%")
    if genre:
        where.append("g.Name LIKE ?")
        params.append(f"%{genre}%")
    if artist:
        where.append("ar.Name LIKE ?")
        params.append(f"%{artist}%")

    if not (search or genre or artist):
        return {
            "error": "no_filters",
            "message": "Provide at least one of: search, genre, or artist.",
        }

    params.append(limit)
    rows = query(
        f"""
        SELECT t.TrackId, t.Name AS Title, t.UnitPrice,
               ar.Name AS Artist, al.Title AS Album,
               COALESCE(g.Name,'Unknown') AS Genre,
               (SELECT COUNT(*) FROM InvoiceLine il WHERE il.TrackId = t.TrackId)
                   AS Popularity
          FROM Track t
          JOIN Album al  ON al.AlbumId  = t.AlbumId
          JOIN Artist ar ON ar.ArtistId = al.ArtistId
          LEFT JOIN Genre g ON g.GenreId = t.GenreId
         WHERE {' AND '.join(where)}
         ORDER BY Popularity DESC, t.TrackId ASC
         LIMIT ?
        """,
        tuple(params),
    )

    return {
        "result_count": len(rows),
        "tracks": [
            {
                "track_id": r["TrackId"],
                "title": r["Title"],
                "artist": r["Artist"],
                "album": r["Album"],
                "genre": r["Genre"],
                "price": str(money(r["UnitPrice"])),
            }
            for r in rows
        ],
    }


@tool
def build_music_cart(
    runtime: ToolRuntime,
    budget: float | None = None,
    genres: list[str] | None = None,
    artists: list[str] | None = None,
    target_tracks: int | None = None,
    min_distinct_artists: int = 1,
    exclude_owned: bool = True,
    replace_existing: bool = True,
) -> dict:
    """Assemble a cart of tracks that satisfies the customer's constraints.

    This is the right tool whenever the customer describes a *set* of music rather
    than one item: a budget, a couple of genres, a track count, "a good mix", "some
    stuff I don't own yet". It solves the constraints exactly and saves the result
    as the customer's cart.

    Read `unmet_constraints` in the result and tell the customer about anything it
    could not satisfy. Never claim a constraint was met if it is listed there, and
    never recalculate the total yourself — use the one returned.

    Args:
        budget: Maximum total spend in dollars. A hard limit; the cart is never over.
        genres: Genres to draw from, e.g. ["Jazz", "Blues"].
        artists: Specific artists to include.
        target_tracks: How many tracks the customer wants.
        min_distinct_artists: Minimum number of different artists, for variety.
        exclude_owned: Leave out tracks the customer has already purchased.
        replace_existing: Replace the current cart. False adds to it instead.
    """
    customer_id = require_customer_id(runtime.context)

    constraints = CartConstraints(
        budget=_to_decimal(budget),
        genres=[g for g in (genres or []) if g],
        artists=[a for a in (artists or []) if a],
        target_tracks=int(target_tracks) if target_tracks else None,
        min_distinct_artists=max(1, int(min_distinct_artists or 1)),
        exclude_owned=bool(exclude_owned),
        exclude_track_ids=set() if replace_existing else set(_load_cart(runtime, customer_id)),
    )

    plan = build_cart(constraints, customer_id)
    new_ids = [i.track_id for i in plan.items]

    if replace_existing:
        track_ids = new_ids
    else:
        track_ids = _load_cart(runtime, customer_id) + new_ids
    track_ids = track_ids[:MAX_CART_ITEMS]
    _save_cart(runtime, customer_id, track_ids)

    result = plan.to_dict()
    result["cart_saved"] = True
    result["cart_track_count"] = len(track_ids)
    if not replace_existing:
        result["cart_total"] = _cart_summary(track_ids)["total"]
    return result


@tool
def view_cart(runtime: ToolRuntime) -> dict:
    """Show what is currently in the customer's cart.

    The cart persists between conversations, so it's worth checking when someone
    returns — they may have left items in it.
    """
    customer_id = require_customer_id(runtime.context)
    summary = _cart_summary(_load_cart(runtime, customer_id))
    if not summary["items"]:
        summary["message"] = "The cart is empty."
    return summary


@tool
def add_tracks_to_cart(track_ids: list[int], runtime: ToolRuntime) -> dict:
    """Add specific tracks to the cart, by track id.

    Args:
        track_ids: Track ids from search_catalog or build_music_cart.
    """
    customer_id = require_customer_id(runtime.context)
    current = _load_cart(runtime, customer_id)
    added = [int(t) for t in track_ids if int(t) not in current]
    updated = (current + added)[:MAX_CART_ITEMS]
    _save_cart(runtime, customer_id, updated)
    summary = _cart_summary(updated)
    summary["added"] = len(added)
    return summary


@tool
def remove_tracks_from_cart(track_ids: list[int], runtime: ToolRuntime) -> dict:
    """Remove specific tracks from the cart, by track id.

    Args:
        track_ids: Track ids currently in the cart.
    """
    customer_id = require_customer_id(runtime.context)
    drop = {int(t) for t in track_ids}
    updated = [t for t in _load_cart(runtime, customer_id) if t not in drop]
    _save_cart(runtime, customer_id, updated)
    summary = _cart_summary(updated)
    summary["removed"] = len(drop)
    return summary


@tool
def checkout_cart(runtime: ToolRuntime) -> dict:
    """Place an order for everything in the cart. Charges the customer.

    This creates a real order and always pauses for human approval first. Confirm
    the contents and total with the customer before calling it.
    """
    customer_id = require_customer_id(runtime.context)
    track_ids = _load_cart(runtime, customer_id)
    if not track_ids:
        return {"ordered": False, "message": "The cart is empty, so there is nothing to order."}

    summary = _cart_summary(track_ids)
    total = Decimal(summary["total"])
    now = datetime.now()

    with write_conn() as conn:
        cust = conn.execute(
            "SELECT Address, City, State, Country, PostalCode FROM Customer "
            "WHERE CustomerId = ?",
            (customer_id,),
        ).fetchone()
        cur = conn.execute(
            "INSERT INTO Invoice (CustomerId, InvoiceDate, BillingAddress, "
            "BillingCity, BillingState, BillingCountry, BillingPostalCode, Total) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                customer_id,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                cust["Address"],
                cust["City"],
                cust["State"],
                cust["Country"],
                cust["PostalCode"],
                float(total),
            ),
        )
        invoice_id = cur.lastrowid
        prices = {i["track_id"]: float(Decimal(i["price"])) for i in summary["items"]}
        conn.executemany(
            "INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity) "
            "VALUES (?, ?, ?, 1)",
            [(invoice_id, tid, prices[tid]) for tid in track_ids if tid in prices],
        )

    _save_cart(runtime, customer_id, [])

    return {
        "ordered": True,
        "order_id": invoice_id,
        "track_count": summary["track_count"],
        "total": summary["total"],
        "message": (
            f"Order #{invoice_id} placed for {summary['track_count']} tracks, "
            f"${summary['total']}. The cart is now empty."
        ),
    }


MERCH_TOOLS = [
    search_catalog,
    build_music_cart,
    view_cart,
    add_tracks_to_cart,
    remove_tracks_from_cart,
    checkout_cart,
]
