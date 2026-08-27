"""
Reading FPL's injury and suspension text into per-gameweek availability.

The model used to compute one availability figure per player and apply it to
every gameweek of the horizon. That writes off anyone out now but back soon -
"Expected back 22 Aug" scored zero for all five gameweeks when the player is fit
from the second - and holds a knock at 75% for five weeks when it will resolve
in one. Both errors run in the direction of ignoring cheap, barely-owned
players, which is where the returning-player edge lives.
"""
from __future__ import annotations

import pathlib
from datetime import date

import pytest

import news

TODAY = date(2026, 8, 18)


@pytest.mark.parametrize("text,expected", [
    ("Groin injury - Expected back 22 Aug", date(2026, 8, 22)),
    ("Suspended until 29 Aug", date(2026, 8, 29)),
    ("Hamstring injury - Expected back 30 Aug", date(2026, 8, 30)),
    ("Leg injury - Expected back 28 Nov", date(2026, 11, 28)),
    ("Knee injury - Expected back 6 Sep", date(2026, 9, 6)),
])
def test_return_dates_are_read(text, expected):
    info = news.parse(text, today=TODAY)
    assert info is not None and info.returns_on == expected
    assert not info.indefinite


def test_a_season_straddles_new_year():
    """
    Dates arrive without a year. In February, "back 10 Jan" is last month and
    "back 10 Dec" is ten months gone - so it must mean the December just past,
    not the one ten months ahead.
    """
    feb = date(2027, 2, 1)
    assert news.parse("Expected back 10 Jan", today=feb).returns_on == date(2027, 1, 10)
    assert news.parse("Expected back 10 Dec", today=feb).returns_on == date(2026, 12, 10)
    aug = date(2026, 8, 18)
    assert news.parse("Expected back 10 Jan", today=aug).returns_on == date(2027, 1, 10)


def test_indefinite_absences_are_distinguished_from_unreadable_ones():
    """
    The two must not be conflated. An unreadable string is how we find out FPL
    changed its templates, and it has to stay visible in the logs.
    """
    assert news.parse("Hamstring injury - Unknown return date", today=TODAY).indefinite
    assert news.parse("Ankle injury - No return date", today=TODAY).indefinite
    assert news.parse("Something entirely new from FPL", today=TODAY) is None


def test_percentage_and_transfer_templates_are_known_but_dateless():
    """
    These are understood and carry no date. Left out of the known set they
    filled the unparsed log with 53 entries a run, which would bury a genuinely
    new template - the one thing that log exists to surface.
    """
    for text in [
        "Knock - 75% chance of playing",
        "Calf injury - 50% chance of playing",
        "Has joined Getafe permanently",
        "Has joined Elche on loan for the rest of the season",
        "Has returned to Getafe CF",
    ]:
        assert news.parse(text, today=TODAY) is None
        assert news.is_known_template(text), f"{text!r} should be a recognised template"

    assert not news.is_known_template("Something entirely new from FPL")


class TestAvailabilityOnADate:
    def test_before_the_return_he_cannot_play(self):
        info = news.parse("Suspended until 29 Aug", today=TODAY)
        assert news.availability_on(info, date(2026, 8, 21), base=0.0, status="s") == 0.0

    def test_on_and_after_the_return_he_can(self):
        info = news.parse("Suspended until 29 Aug", today=TODAY)
        # The status flag still reads 's' because FPL has not cleared it; the
        # news carries the better information and must win.
        assert news.availability_on(info, date(2026, 8, 29), base=0.0, status="s") == 1.0
        assert news.availability_on(info, date(2026, 9, 4), base=0.0, status="s") == 1.0

    def test_an_indefinite_absence_never_returns_him(self):
        info = news.parse("Hamstring injury - Unknown return date", today=TODAY)
        for when in (date(2026, 9, 1), date(2026, 12, 1)):
            assert news.availability_on(info, when, base=0.0, status="i") == 0.0

    def test_no_news_leaves_the_caller_alone(self):
        assert news.availability_on(None, date(2026, 9, 1), base=0.75, status="d") == 0.75


def test_a_doubt_fades_but_an_absence_does_not():
    """
    A 75% knock is not 75% in five weeks. But decaying from zero would quietly
    restore a long-term injury to near-full fitness by gameweek four, so the
    caller must only apply this to genuine doubts.
    """
    assert news.decay_doubt(0.75, 0) == pytest.approx(0.75)
    assert news.decay_doubt(0.75, 1) > 0.75
    assert news.decay_doubt(0.75, 4) > news.decay_doubt(0.75, 1)
    assert news.decay_doubt(0.75, 10) <= 1.0
    assert news.decay_doubt(1.0, 5) == pytest.approx(1.0)


def test_a_returning_player_is_eased_back_in():
    """Fit to play is not the same as straight back to ninety minutes."""
    info = news.parse("Expected back 22 Aug", today=TODAY)
    first = news.ramp_multiplier(info, date(2026, 8, 22))
    second = news.ramp_multiplier(info, date(2026, 8, 30))
    later = news.ramp_multiplier(info, date(2026, 10, 1))
    assert first < second < later == 1.0


def test_a_short_absence_needs_no_ramp():
    """A one-match suspension costs no match fitness."""
    info = news.parse("Suspended until 29 Aug", today=TODAY)
    assert news.ramp_multiplier(info, date(2026, 8, 29), absence_days=7) == 1.0


# ── integration: the model actually uses this ──────────────────────────────

@pytest.fixture(scope="module")
def scored():
    """Score the committed snapshot, with news applied per gameweek."""
    from pathlib import Path
    import priors
    import xp_model as X

    snaps = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    bs_files = sorted(snaps.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not bs_files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(bs_files[0])
    fixtures = X.load_snapshot(sorted(snaps.glob("fixtures_*.json.gz"), reverse=True)[0])
    ps = priors.build_priors()
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
    events = X.next_events(bootstrap, 5)
    return model.expected_points(events), events, bootstrap, model


def test_a_player_returning_mid_horizon_is_worth_nothing_now_and_something_later(scored):
    """
    The behaviour the whole module exists for. Someone out for gameweek one but
    back for gameweek three must score zero now and a real number later - not
    zero throughout, which is what a single flat availability produced.
    """
    frame, events, bootstrap, model = scored
    ids = {int(e["id"]): e for e in bootstrap["elements"]}

    # Selection has to be on the player's own FIRST FIXTURE, not on the
    # gameweek deadline. A gameweek runs Friday to Monday, and someone whose
    # return date lands on his team's Saturday kickoff is available for that
    # gameweek - Garner returns 22 Aug and Everton play on 22 Aug. Selecting on
    # the deadline swept those players in and then asserted they score zero,
    # which encoded the anchor bug as the expected behaviour.
    first_kickoffs = model._return_dates_by_team(events[0])

    returning = []
    for _, row in frame.iterrows():
        info = news.parse((ids[int(row["id"])].get("news") or ""))
        if info and info.returns_on and row["status"] in ("i", "s"):
            kick = first_kickoffs.get(int(row["team"]))
            if kick is not None and kick >= info.returns_on:
                continue                      # genuinely back for gameweek one
            first, last = row[f"xp_gw{events[0]}"], row[f"xp_gw{events[-1]}"]
            if last > 0.5:
                returning.append((row["web_name"], first, last))

    assert returning, "snapshot contains nobody returning inside the horizon"
    for name, first, last in returning:
        assert first == pytest.approx(0.0, abs=1e-9), f"{name} should score nothing while out"
        assert last > first, f"{name} should be worth more once he is back"


def test_an_indefinite_absence_scores_zero_across_the_whole_horizon(scored):
    frame, events, bootstrap, model = scored
    ids = {int(e["id"]): e for e in bootstrap["elements"]}
    checked = 0
    for _, row in frame.iterrows():
        info = news.parse((ids[int(row["id"])].get("news") or ""))
        if info and info.indefinite and row["status"] == "i":
            for ev in events:
                assert row[f"xp_gw{ev}"] == pytest.approx(0.0, abs=1e-9), (
                    f"{row['web_name']} is out indefinitely but scores in GW{ev}"
                )
            checked += 1
    assert checked, "snapshot contains no indefinite absences"


def test_a_flagged_doubt_recovers_across_the_horizon(scored):
    """
    A knock resolves. Asserted on availability rather than xP, because xP also
    carries fixture difficulty and would confound the thing being measured.
    """
    import priors
    import xp_model as X

    _, events, bootstrap, _model = scored
    snaps = pathlib.Path(__file__).resolve().parent.parent / "data" / "snapshots"
    fixtures = X.load_snapshot(sorted(snaps.glob("fixtures_*.json.gz"), reverse=True)[0])
    model = X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=5))

    first = model._availability(events[0])
    last = model._availability(events[-1])
    frame = model.players

    doubts = frame.index[(frame["status"] == "d") & frame["chance_of_playing_next_round"].between(1, 99)]
    assert len(doubts), "snapshot contains no flagged doubts"
    for i in doubts:
        assert last[i] > first[i], (
            f"{frame['web_name'][i]} is a short-term doubt but is no more available "
            f"in GW{events[-1]} than in GW{events[0]}"
        )
        assert last[i] <= 1.0

    # And the opposite guarantee: an indefinite absence must never recover.
    out = frame.index[(frame["status"] == "i") & (frame["news"].fillna("").str.contains("Unknown return"))]
    for i in out:
        assert last[i] == pytest.approx(0.0), (
            f"{frame['web_name'][i]} is out indefinitely but recovered to {last[i]:.2f}"
        )
