"""
Fixes for findings from two independent audits, 2026-10-12.

Each test here corresponds to a defect that was live in production, and every
one of them was invisible to the suite that existed at the time.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import deadline_state as ds
import rivals
import watchdog

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

SEASON = [
    {"id": 5, "deadline_time": "2026-09-18T17:30:00Z"},
    {"id": 6, "deadline_time": "2026-10-10T10:00:00Z"},
    {"id": 7, "deadline_time": "2026-10-17T10:00:00Z"},
]
GW6_DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)


def _code(name: str) -> str:
    lines = [l for l in (WORKFLOWS / name).read_text().splitlines()
             if not l.strip().startswith("#")]
    return " ".join(" ".join(lines).split())


# ------------------------------------------------- 1. the token race

def test_the_weekly_run_cannot_overlap_itself():
    """
    The marker is an unsynchronised read-then-act, so it closes the sequential
    case only. Two concurrent runs both spend the same single-use refresh token
    and then race to store their replacement; if the loser writes last, the
    stored token is already invalid and every later run is locked out until a
    human issues a new one by hand.
    """
    code = _code("weekly_run.yml")
    assert "concurrency:" in code, "nothing stops two weekly runs overlapping"
    assert "group: fpl-weekly-run" in code
    assert "cancel-in-progress: false" in code, (
        "cancelling a run that may be mid-submission is worse than queueing"
    )


# --------------------------------------- 2. the watchdog's blind spot

def test_a_passed_deadline_with_nothing_submitted_is_an_alert():
    """
    Production, 2026-10-10T11:34Z: ninety minutes after the GW6 deadline passed
    with nothing submitted, the watchdog logged `waiting - GW7 deadline is
    166.4h away`, alert False. assess() keyed off next_event(), which only ever
    looks forward, so a missed gameweek stopped existing the instant its
    deadline passed.
    """
    v = watchdog.assess(SEASON, last_submitted_gw=5,
                        now=GW6_DEADLINE + timedelta(hours=1.5))

    assert v["state"] == "missed"
    assert v["alert"] is True
    assert v["missed_event"] == 6
    assert v["trigger"] is False, "nothing can be submitted for a passed deadline"


def test_a_stale_miss_stops_alerting_but_never_reports_an_all_clear():
    """
    A miss is permanent, so alerting forever is just noise - but reporting
    health over a dropped gameweek is the one thing this must never do.

    So the state stays forward-looking and `missed_event` carries the fact,
    which is what suppresses the all-clear. A miss must never pre-empt the
    verdict on the gameweek ahead, because that one can still be remediated.
    """
    # +100h: past the 36h alerting window, but still before GW7's deadline, so
    # GW6 is the most recently passed gameweek. (At +200h GW7 has also passed
    # and is correctly reported as the newer miss.)
    v = watchdog.assess(SEASON, last_submitted_gw=5,
                        now=GW6_DEADLINE + timedelta(hours=100))
    assert v["alert"] is False
    assert v["missed_event"] == 6, "the miss must survive into the verdict"
    assert "GW6" in v["detail"] and "nothing submitted" in v["detail"]


def test_the_all_clear_is_suppressed_while_a_miss_is_outstanding(monkeypatch):
    """The 72h heartbeat would otherwise narrate health over the dropped week."""
    sent = []

    class _Gh:
        def __init__(self): self.vars = {watchdog.MARKER_VARIABLE: "5"}
        def get_variable(self, n): return self.vars.get(n)
        def set_variable(self, n, v): self.vars[n] = v; return True
        def dispatch_workflow(self, *a, **k): return True

    class _N:
        def send(self, t): sent.append(t); return True

    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    watchdog.run(now=GW6_DEADLINE + timedelta(hours=100), gh=_Gh(), notifier=_N(),
                 fetcher=lambda url, timeout=30: {"events": SEASON})

    assert not any("All clear" in m for m in sent), "all-clear sent over a missed gameweek"


def test_a_miss_never_suppresses_remediation_of_the_next_deadline():
    """
    The bug this restructure fixed. An earlier version returned early on a
    miss, so once GW6 was dropped the watchdog would have reported `missed`
    and `trigger: False` for every later gameweek - silently disabling
    remediation for the rest of the season. Two existing tests caught it.
    """
    # GW6 missed (marker 5), and GW7's deadline is now 3h away, unsubmitted.
    gw7 = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)
    v = watchdog.assess(SEASON, last_submitted_gw=5, now=gw7 - timedelta(hours=3))

    assert v["state"] == "overdue", "the upcoming deadline must keep precedence"
    assert v["trigger"] is True, "a past miss must not disable remediation"
    assert v["missed_event"] == 6, "but the miss must still be reported"
    assert "GW6" in v["detail"]


def test_a_submitted_gameweek_is_not_reported_as_missed():
    """Guards the other direction: alerting every week would train you to ignore it."""
    v = watchdog.assess(SEASON, last_submitted_gw=6,
                        now=GW6_DEADLINE + timedelta(hours=1.5))
    assert v["state"] != "missed"
    assert v["missed_event"] is None


def test_previous_event_is_actually_wired_in():
    """
    `previous_deadline()` existed for exactly this and `grep` found one hit in
    the whole repo: its own definition. A helper nobody calls is not a feature.
    """
    assert ds.previous_event(SEASON, GW6_DEADLINE + timedelta(hours=1))["id"] == 6
    assert ds.previous_deadline(SEASON, GW6_DEADLINE + timedelta(hours=1)) == GW6_DEADLINE
    assert ds.previous_event(SEASON, datetime(2026, 1, 1, tzinfo=timezone.utc)) is None


# ------------------------------------------- 3. the disconnected alarm

def test_the_step_that_runs_manager_can_raise_the_alarm():
    """
    manager.py alerts when it submits a team and then cannot write the marker -
    the case where the next tick resubmits with its transfers spent and a -4
    available. The step had no Telegram vars, so that degraded to a
    NullNotifier and the run exited 0. Its unit test monkeypatched from_env,
    patching away the exact thing that was broken.

    Parsed as text, not with pyyaml: pyyaml is not in requirements.txt, so
    importing it here passed locally and died in CI with ModuleNotFoundError -
    which is why test_workflow_invariants.py already avoids it.
    """
    text = (WORKFLOWS / "weekly_run.yml").read_text()

    marker = "- name: Run FPL Auto Manager"
    assert marker in text, "the step that runs manager.py was renamed"
    block = text[text.index(marker):]
    nxt = block.find("\n      - name:", 1)
    block = block[:nxt] if nxt > 0 else block

    assert "manager.py" in block, "wrong block located"
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert var in block, (
            f"the step running manager.py has no {var}, so manager.py cannot "
            "send its own alert and degrades to a NullNotifier"
        )


# ------------------------------------------------ 4. the fan-out

def test_the_fan_out_stops_when_its_budget_is_spent():
    """
    100 sequential requests at RETRIES=3 is a ~162 minute worst case against a
    15 minute job timeout: nine slow URLs out of a hundred cost the gameweek
    without a submission ever being attempted.
    """
    fetched = []
    fake_now = [0.0]

    def opener(url):
        fetched.append(url)
        if "/picks/" in url:
            fake_now[0] += 10.0          # each rival "takes" ten seconds
            return {"picks": [{"element": 1, "multiplier": 1}]}
        if "/entry/" in url:
            return {"leagues": {"classic": [{"id": 1, "name": "L", "rank_count": 50}]}}
        return {"standings": {"results": [{"entry": i} for i in range(100, 200)]}}

    client = rivals.PublicFPL(opener=opener)
    fo = rivals.field_ownership(client, 5413589, 6, budget_s=55.0,
                                clock=lambda: fake_now[0])

    picks = [u for u in fetched if "/picks/" in u]
    # 55s of budget at 10s a rival stops after ~6. The loose bound this
    # originally used (< 99) was satisfied even with the budget removed,
    # because rival_ids caps each league at 50 - the mutation was invisible.
    assert len(picks) <= 8, f"the budget did not stop the fan-out ({len(picks)} fetched)"
    assert fo.managers == len(picks), "should keep the rivals it did read"


def test_the_fan_out_retries_sparingly():
    """Three retries with exponential backoff is a 97s worst case per URL."""
    assert rivals.PICKS_RETRIES < rivals.RETRIES
    assert rivals.REQUEST_DELAY > 0, (
        "config.STRATEGY records that bursts get rate limited; this path fires "
        "a hundred requests back to back"
    )


def test_a_thin_field_falls_back_to_the_template(monkeypatch):
    """
    With one readable rival every owned player reads as 100% owned, which at
    the shipped weight swings effective xP by about +-15% on one stranger's
    squad. A worse target beats a garbage distribution.
    """
    monkeypatch.setattr(rivals, "field_ownership",
                        lambda *a, **k: rivals.FieldOwnership(
                            event=6, managers=3, squad={1: 1.0}, captain={}))
    assert rivals.field_ownership_or_none(5413589, 6) is None


def test_a_full_field_is_still_used(monkeypatch):
    """Guards the other direction: a permanently-falling-back tilt is inert."""
    monkeypatch.setattr(rivals, "field_ownership",
                        lambda *a, **k: rivals.FieldOwnership(
                            event=6, managers=94, squad={1: 0.84}, captain={1: 0.42}))
    fo = rivals.field_ownership_or_none(5413589, 6)
    assert fo is not None and fo.managers == 94


def test_the_floor_is_below_the_real_field_size():
    """94 rivals today; a floor above that would silently disable the feature."""
    assert 2 <= rivals.MIN_RIVALS <= 40
