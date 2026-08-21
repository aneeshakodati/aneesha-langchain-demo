"""Cart constraint solver.

The budget assertions are the point. A cart that comes in over budget is the
failure customers actually notice, and it's the one thing a language model
genuinely cannot be trusted to check.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chinook_support.cart import (
    CartConstraints,
    build_cart,
    resolve_artists,
    resolve_genres,
)
from chinook_support.db import owned_track_ids

CUSTOMER = 1


def total_of(plan) -> Decimal:
    """Recompute independently rather than trusting the reported total."""
    return sum((item.unit_price for item in plan.items), Decimal("0.00"))


@pytest.mark.parametrize("budget", ["5.00", "15.00", "40.00"])
def test_never_exceeds_budget(budget):
    plan = build_cart(
        CartConstraints(budget=Decimal(budget), genres=["Jazz", "Blues"], target_tracks=40),
        CUSTOMER,
    )
    assert total_of(plan) <= Decimal(budget)
    assert plan.total == total_of(plan)


def test_reported_total_matches_the_items():
    plan = build_cart(CartConstraints(genres=["Rock"], target_tracks=15), CUSTOMER)
    assert plan.total == total_of(plan)
    assert Decimal(plan.to_dict()["total"]) == total_of(plan)


def test_excludes_tracks_the_customer_already_owns():
    owned = owned_track_ids(CUSTOMER)
    assert owned, "fixture customer should have purchase history"
    plan = build_cart(
        CartConstraints(genres=["Rock"], target_tracks=40, exclude_owned=True), CUSTOMER
    )
    assert not {item.track_id for item in plan.items} & owned


def test_exclude_owned_can_be_turned_off():
    plan = build_cart(
        CartConstraints(genres=["Rock"], target_tracks=40, exclude_owned=False), CUSTOMER
    )
    assert plan.items


def test_honours_minimum_distinct_artists():
    plan = build_cart(
        CartConstraints(genres=["Rock"], target_tracks=10, min_distinct_artists=5), CUSTOMER
    )
    assert plan.distinct_artists >= 5
    assert not plan.unmet


def test_only_returns_requested_genres():
    plan = build_cart(CartConstraints(genres=["Jazz"], target_tracks=10), CUSTOMER)
    assert set(plan.genre_breakdown) == {"Jazz"}


def test_multi_genre_requests_stay_mixed():
    """Round-robin fill should not return 12 Rock tracks and one Jazz."""
    plan = build_cart(
        CartConstraints(genres=["Jazz", "Blues"], target_tracks=12, min_distinct_artists=1),
        CUSTOMER,
    )
    assert len(plan.genre_breakdown) == 2
    smallest = min(plan.genre_breakdown.values())
    assert smallest >= 2, plan.genre_breakdown


def test_unsatisfiable_constraints_are_reported_not_hidden():
    """The solver must say what it could not do -- silence here is the real bug."""
    plan = build_cart(
        CartConstraints(
            budget=Decimal("2.00"), genres=["Jazz"], target_tracks=10, min_distinct_artists=5
        ),
        CUSTOMER,
    )
    assert plan.unmet, "expected unmet constraints to be reported"
    assert not plan.to_dict()["all_constraints_satisfied"]
    # Budget is still respected even when everything else fails.
    assert total_of(plan) <= Decimal("2.00")


def test_unknown_genre_is_reported_rather_than_silently_dropped():
    plan = build_cart(CartConstraints(genres=["Polka"], target_tracks=5), CUSTOMER)
    assert any("Polka" in note for note in plan.unmet)


def test_is_deterministic():
    constraints = CartConstraints(
        budget=Decimal("15.00"), genres=["Jazz", "Blues"], target_tracks=12
    )
    first = build_cart(constraints, CUSTOMER)
    second = build_cart(constraints, CUSTOMER)
    assert [i.track_id for i in first.items] == [i.track_id for i in second.items]


def test_genre_and_artist_resolution_is_fuzzy_but_honest():
    matched, unmatched = resolve_genres(["jazz", "blues", "polka"])
    assert "Jazz" in matched and "Blues" in matched
    assert unmatched == ["polka"]

    artist_ids, missing = resolve_artists(["AC/DC", "Definitely Not A Band"])
    assert artist_ids
    assert missing == ["Definitely Not A Band"]
