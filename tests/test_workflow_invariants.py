"""
Properties of the workflow files that no Python test would otherwise catch.

Every entry in here is a failure that has actually happened to this repo or was
one merge away from happening. Workflows are the part nobody runs locally.
"""
from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Read as text rather than parsed YAML on purpose. pyyaml is not in
# requirements.txt, so `importorskip` would make every assertion here vanish in
# CI - and a test that silently stops running is this repo's most-repeated
# failure, not a hypothetical one. These are all substring properties, so the
# parser buys nothing.


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def _code(name: str) -> str:
    """
    The workflow with comments stripped and whitespace collapsed.

    Stripping comments is the point: an earlier version of this file matched
    against the raw text and passed happily when the condition was deleted,
    because the word it was looking for also appeared in the comment explaining
    the condition. A test that reads prose is not a test.
    """
    lines = [l for l in _text(name).splitlines() if not l.strip().startswith("#")]
    return " ".join(" ".join(lines).split())


def test_the_gate_cannot_silently_cancel_the_weekly_run():
    """
    `needs: gate` alone skips the dependent job whenever the gate errors, so a
    crash in a cheap optimisation step would cost a whole gameweek without
    failing anything. The gate is not what enforces the window - manager.py
    re-checks it - so this must fail OPEN.
    """
    # Scoped to this job's own condition. Matching the whole file passed when
    # the condition was deleted, because `always()` also appears on the
    # "Upload logs" step - the second time a loose match here proved nothing.
    body = _code("weekly_run.yml")
    start = body.index("manage-team:")
    cond = body[start:body.index("steps:", start)]

    assert "always()" in cond, (
        "without always(), a failed gate skips manage-team before the `if` is "
        "evaluated, and the deadline passes with no run at all"
    )
    assert "!= 'false'" in cond, (
        "the condition must run unless the gate said 'false' explicitly; "
        "`== 'true'` treats a crashed gate as a decision not to play"
    )


def test_the_weekly_run_samples_often_enough_to_have_a_retry():
    """
    A 24h-wide window sampled every 24h admitted exactly one tick per gameweek
    for all 38, so a single dropped run missed the deadline. That is what
    happened to GW2.
    """
    import deadline_state as ds

    assert "- cron: '0 * * * *'" in _text("weekly_run.yml"), (
        "expected an hourly cron on the weekly run"
    )

    band = ds.DEADLINE_WINDOW_MAX - ds.DEADLINE_WINDOW_MIN
    assert band >= 2.0, (
        f"an hourly cron on a {band}h band leaves too few qualifying ticks for "
        "a dropped run to be survivable"
    )


def test_the_watchdog_has_its_own_schedule():
    """
    deadline_check.yml was workflow_dispatch-only, triggered by a scheduler
    that had been disabled since April. A safety net wired to the thing it is
    watching dies with it, silently, for months.
    """
    text = _text("watchdog.yml")
    assert "schedule:" in text, "the watchdog must not depend on anything else to fire"
    assert "cron:" in text, "a schedule block with no cron fires nothing"


def test_the_watchdog_can_dispatch_the_run_it_is_watching():
    """Alerting without remediation is useless when nobody is at a laptop."""
    assert "actions: write" in _text("watchdog.yml"), (
        "the watchdog needs actions:write to dispatch the weekly run"
    )


def test_the_watchdog_installs_nothing():
    """
    It has to survive the failures it reports on, including a broken dependency
    install. It already died once of ModuleNotFoundError, taking its own
    failure alert down with it.
    """
    assert "pip install" not in _code("watchdog.yml"), (
        "watchdog.yml installs dependencies; keep the import graph stdlib-only "
        "instead (see test_notify.py)"
    )
