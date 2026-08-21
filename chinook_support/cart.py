"""Constraint-based cart builder — pure functions, no LLM.

The business problem: *"build me a cart of jazz and blues under $15, nothing I
already own, spread across at least 3 artists."*

That is a constraint-satisfaction problem over 3,503 tracks, and it is exactly the
kind of thing language models are bad at. Ask a model to pick tracks under a budget
and it will produce a plausible-looking list whose prices don't add up. So the
division of labour here is:

    model  -> parses intent into `CartConstraints`
    python -> solves it, exactly, and reports what it could not satisfy

The last part matters as much as the solve. `CartPlan.unmet` is an explicit list of
constraints the solver could *not* honour. A solver that silently returns a $22 cart
when you asked for $15 is worse than useless; this one hands the model a fact it
must relay. Reliability is mostly about making failure legible.

Selection is fully deterministic (popularity desc, then track id) so the same
request yields the same cart — which is what makes
`evals/evaluators.py::cart_constraints_satisfied` a stable regression test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .config import MAX_CART_ITEMS
from .db import money, owned_track_ids, query


@dataclass
class CartConstraints:
    """What the customer asked for, normalized."""

    budget: Decimal | None = None
    genres: list[str] = field(default_factory=list)
    artists: list[str] = field(default_factory=list)
    target_tracks: int | None = None
    min_distinct_artists: int = 1
    exclude_owned: bool = True
    exclude_track_ids: set[int] = field(default_factory=set)


@dataclass
class CartItem:
    track_id: int
    title: str
    artist: str
    album: str
    genre: str
    unit_price: Decimal

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "price": str(self.unit_price),
        }


@dataclass
class CartPlan:
    items: list[CartItem]
    total: Decimal
    constraints: CartConstraints
    unmet: list[str] = field(default_factory=list)

    @property
    def distinct_artists(self) -> int:
        return len({i.artist for i in self.items})

    @property
    def genre_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.genre] = counts.get(item.genre, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "items": [i.to_dict() for i in self.items],
            "track_count": len(self.items),
            "total": str(self.total),
            "budget": str(self.constraints.budget) if self.constraints.budget else None,
            "distinct_artists": self.distinct_artists,
            "genre_breakdown": self.genre_breakdown,
            "unmet_constraints": self.unmet,
            "all_constraints_satisfied": not self.unmet,
        }


# --- Fuzzy resolution of user-supplied names ---------------------------------


def resolve_genres(names: list[str]) -> tuple[list[str], list[str]]:
    """Map loose user strings ('jazz', 'r&b') onto real Genre rows.

    Returns (matched_genre_names, unmatched_inputs). Unmatched inputs become
    entries in `CartPlan.unmet` rather than being silently dropped.
    """
    if not names:
        return [], []
    matched: list[str] = []
    unmatched: list[str] = []
    for name in names:
        rows = query(
            "SELECT Name FROM Genre WHERE LOWER(Name) = LOWER(?) "
            "OR LOWER(Name) LIKE LOWER(?) ORDER BY LENGTH(Name) LIMIT 3",
            (name, f"%{name}%"),
        )
        if rows:
            matched.extend(r["Name"] for r in rows if r["Name"] not in matched)
        else:
            unmatched.append(name)
    return matched, unmatched


def resolve_artists(names: list[str]) -> tuple[list[int], list[str]]:
    """Map loose artist strings onto ArtistIds."""
    if not names:
        return [], []
    matched: list[int] = []
    unmatched: list[str] = []
    for name in names:
        rows = query(
            "SELECT ArtistId, Name FROM Artist WHERE LOWER(Name) = LOWER(?) "
            "OR LOWER(Name) LIKE LOWER(?) ORDER BY LENGTH(Name) LIMIT 2",
            (name, f"%{name}%"),
        )
        if rows:
            matched.extend(r["ArtistId"] for r in rows if r["ArtistId"] not in matched)
        else:
            unmatched.append(name)
    return matched, unmatched


# --- Candidate retrieval -----------------------------------------------------


def _candidates(
    genres: list[str],
    artist_ids: list[int],
    excluded: set[int],
    max_unit_price: Decimal | None,
) -> list[CartItem]:
    """Fetch eligible tracks, ordered by popularity then id.

    'Popularity' is how many times a track appears across all InvoiceLines — the
    store's own sales signal, which is a better default ranking than alphabetical
    and gives the recommendations a "customers also bought" flavour for free.
    """
    where: list[str] = ["1=1"]
    params: list = []

    if genres:
        where.append(f"g.Name IN ({','.join('?' * len(genres))})")
        params.extend(genres)
    if artist_ids:
        where.append(f"ar.ArtistId IN ({','.join('?' * len(artist_ids))})")
        params.extend(artist_ids)
    if max_unit_price is not None:
        where.append("t.UnitPrice <= ?")
        params.append(float(max_unit_price))

    rows = query(
        f"""
        SELECT t.TrackId, t.Name AS Title, t.UnitPrice,
               ar.Name AS Artist, al.Title AS Album,
               COALESCE(g.Name, 'Unknown') AS Genre,
               (SELECT COUNT(*) FROM InvoiceLine il WHERE il.TrackId = t.TrackId)
                   AS Popularity
          FROM Track t
          JOIN Album al  ON al.AlbumId  = t.AlbumId
          JOIN Artist ar ON ar.ArtistId = al.ArtistId
          LEFT JOIN Genre g ON g.GenreId = t.GenreId
         WHERE {' AND '.join(where)}
         ORDER BY Popularity DESC, t.TrackId ASC
         LIMIT 2000
        """,
        tuple(params),
    )

    return [
        CartItem(
            track_id=r["TrackId"],
            title=r["Title"],
            artist=r["Artist"],
            album=r["Album"],
            genre=r["Genre"],
            unit_price=money(r["UnitPrice"]),
        )
        for r in rows
        if r["TrackId"] not in excluded
    ]


# --- The solver --------------------------------------------------------------


def build_cart(constraints: CartConstraints, customer_id: int) -> CartPlan:
    """Select a set of tracks satisfying `constraints` as far as possible.

    Strategy, in two passes:

      Pass 1 (diversity): take the single most popular eligible track from each
        distinct artist, in artist-popularity order, until `min_distinct_artists`
        is met. Doing diversity first guarantees it isn't crowded out by a run of
        tracks from one blockbuster album.

      Pass 2 (fill): add remaining tracks in popularity order, round-robining
        across the requested genres so a multi-genre request stays balanced
        instead of returning 12 Rock tracks and 1 Jazz.

    Both passes respect the budget as a hard constraint — the cart is never over.
    """
    excluded = set(constraints.exclude_track_ids)
    if constraints.exclude_owned:
        excluded |= owned_track_ids(customer_id)

    genres, bad_genres = resolve_genres(constraints.genres)
    artist_ids, bad_artists = resolve_artists(constraints.artists)

    unmet: list[str] = []
    for name in bad_genres:
        unmet.append(f"No genre matching {name!r} exists in the catalog.")
    for name in bad_artists:
        unmet.append(f"No artist matching {name!r} exists in the catalog.")

    pool = _candidates(genres, artist_ids, excluded, constraints.budget)
    if not pool:
        unmet.append("No tracks matched the requested filters.")
        return CartPlan(items=[], total=Decimal("0.00"), constraints=constraints, unmet=unmet)

    budget = constraints.budget
    limit = min(constraints.target_tracks or MAX_CART_ITEMS, MAX_CART_ITEMS)

    selected: list[CartItem] = []
    total = Decimal("0.00")
    taken: set[int] = set()

    def can_afford(item: CartItem) -> bool:
        return budget is None or (total + item.unit_price) <= budget

    # Pass 1 — one track per artist, for diversity.
    by_artist: dict[str, list[CartItem]] = {}
    for item in pool:
        by_artist.setdefault(item.artist, []).append(item)

    for artist in by_artist:
        if len({i.artist for i in selected}) >= constraints.min_distinct_artists:
            break
        if len(selected) >= limit:
            break
        item = by_artist[artist][0]
        if can_afford(item):
            selected.append(item)
            taken.add(item.track_id)
            total += item.unit_price

    # Pass 2 — fill, round-robining across genres to keep the mix balanced.
    by_genre: dict[str, list[CartItem]] = {}
    for item in pool:
        if item.track_id not in taken:
            by_genre.setdefault(item.genre, []).append(item)

    cursors = {genre: 0 for genre in by_genre}
    exhausted = False
    while len(selected) < limit and not exhausted and by_genre:
        exhausted = True
        for genre, items in by_genre.items():
            if len(selected) >= limit:
                break
            idx = cursors[genre]
            while idx < len(items):
                item = items[idx]
                idx += 1
                if item.track_id in taken:
                    continue
                if can_afford(item):
                    selected.append(item)
                    taken.add(item.track_id)
                    total += item.unit_price
                    exhausted = False
                    break
            cursors[genre] = idx

    # Report anything we could not honour.
    if constraints.target_tracks and len(selected) < constraints.target_tracks:
        unmet.append(
            f"Asked for {constraints.target_tracks} tracks but only {len(selected)} "
            f"fit the other constraints"
            + (f" within the ${budget} budget." if budget else ".")
        )
    distinct = len({i.artist for i in selected})
    if distinct < constraints.min_distinct_artists:
        unmet.append(
            f"Asked for at least {constraints.min_distinct_artists} different "
            f"artists but only {distinct} could be included."
        )

    return CartPlan(
        items=selected,
        total=total.quantize(Decimal("0.01")),
        constraints=constraints,
        unmet=unmet,
    )
