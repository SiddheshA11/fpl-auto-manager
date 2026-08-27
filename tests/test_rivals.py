"""
Field ownership from the people you actually play against.

Every request here is mocked. The point of the module is that it needs no
credentials, and a test that reached the network would be both flaky and a
standing invitation to hit the authenticated endpoints by accident.
"""
from __future__ import annotations

import pytest

import rivals


def _fake(pages: dict):
    """An opener that serves canned JSON and records what was asked for."""
    calls = []

    def opener(url: str):
        calls.append(url)
        for suffix, payload in pages.items():
            if url.endswith(suffix):
                return payload
        return None

    opener.calls = calls
    return opener


ENTRY = {"leagues": {"classic": [
    {"id": 314, "name": "Overall", "rank_count": 8903396},
    {"id": 249, "name": "USA", "rank_count": 329476},
    {"id": 1178688, "name": "Mini", "rank_count": 19},
    {"id": 858098, "name": "Other", "rank_count": 11},
]}}

STANDINGS = {"standings": {"results": [
    {"entry": 5413589}, {"entry": 111}, {"entry": 222}, {"entry": 333},
]}}

def _picks(elements, captain):
    return {"picks": [{"element": e, "multiplier": 2 if e == captain else 1}
                      for e in elements]}


def test_global_leagues_are_not_a_field():
    """
    Overall and country leagues carry millions of entries. They ARE the
    template, so treating them as rivals would reintroduce the exact proxy this
    module exists to replace.
    """
    c = rivals.PublicFPL(_fake({"/entry/5413589/": ENTRY}))
    assert rivals.mini_league_ids(c, 5413589) == [1178688, 858098]


def test_the_manager_is_not_his_own_rival():
    c = rivals.PublicFPL(_fake({"/standings/": STANDINGS}))
    assert 5413589 not in rivals.rival_ids(c, 1178688, 5413589)
    assert sorted(rivals.rival_ids(c, 1178688, 5413589)) == [111, 222, 333]


def test_shares_are_over_managers_not_picks():
    c = rivals.PublicFPL(_fake({
        "/entry/5413589/": ENTRY,
        "/standings/": STANDINGS,
        "/entry/111/event/1/picks/": _picks([1, 2, 3], captain=1),
        "/entry/222/event/1/picks/": _picks([1, 2, 4], captain=2),
        "/entry/333/event/1/picks/": _picks([1, 5, 6], captain=1),
    }))
    f = rivals.field_ownership(c, 5413589, 1)
    assert f.managers == 3
    assert f.squad_share(1) == pytest.approx(1.0)
    assert f.squad_share(2) == pytest.approx(2 / 3)
    assert f.squad_share(99) == 0.0


def test_captaincy_is_not_squad_ownership():
    """
    The distinction the whole rank-aware idea rests on. Measured on GW1,
    João Pedro was owned by 77.8% of one league and captained by 16.7%.
    """
    c = rivals.PublicFPL(_fake({
        "/entry/5413589/": ENTRY,
        "/standings/": STANDINGS,
        "/entry/111/event/1/picks/": _picks([1, 2, 3], captain=1),
        "/entry/222/event/1/picks/": _picks([1, 2, 4], captain=2),
        "/entry/333/event/1/picks/": _picks([1, 2, 6], captain=1),
    }))
    f = rivals.field_ownership(c, 5413589, 1)
    assert f.squad_share(2) == pytest.approx(1.0)
    assert f.captain_share(2) == pytest.approx(1 / 3)
    assert f.captain_share(1) == pytest.approx(2 / 3)


def test_a_gameweek_with_no_picks_yields_nothing_rather_than_raising():
    """
    /picks/ publishes nothing before a gameweek completes. The weekly run must
    fall back to the template, not crash.
    """
    c = rivals.PublicFPL(_fake({"/entry/5413589/": ENTRY, "/standings/": STANDINGS}))
    f = rivals.field_ownership(c, 5413589, 7)
    assert f.managers == 0
    assert f.squad_share(1) == 0.0


def test_one_dead_rival_does_not_lose_the_rest():
    """A manager who never set a squad 404s. That is data, not a failure."""
    c = rivals.PublicFPL(_fake({
        "/entry/5413589/": ENTRY,
        "/standings/": STANDINGS,
        "/entry/111/event/1/picks/": _picks([1, 2], captain=1),
        "/entry/333/event/1/picks/": _picks([1, 3], captain=1),
    }))
    f = rivals.field_ownership(c, 5413589, 1)
    assert f.managers == 2
    assert f.squad_share(1) == pytest.approx(1.0)


def test_nothing_here_touches_an_authenticated_endpoint():
    """
    The reason this module exists rather than LeagueAnalyzer: that one calls
    get_my_leagues() through the authenticated client, and the refresh token is
    single-use. Spending it locks the account out.
    """
    opener = _fake({
        "/entry/5413589/": ENTRY,
        "/standings/": STANDINGS,
        "/entry/111/event/1/picks/": _picks([1], captain=1),
    })
    rivals.field_ownership(rivals.PublicFPL(opener), 5413589, 1)
    assert opener.calls, "expected some requests"
    for url in opener.calls:
        assert "/me/" not in url and "/my-team/" not in url, url
        assert "login" not in url and "token" not in url, url
