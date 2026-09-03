"""
Put the repo root on sys.path so `pytest tests/` works, not only
`python3 -m pytest tests/`.

Without this the bare `pytest` invocation fails to import `manager` and friends,
which is a live footgun now that CI runs the suite: the obvious command is the
one that breaks.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


import pytest


@pytest.fixture(autouse=True)
def _no_rival_network(monkeypatch):
    """
    Nothing in the suite may fetch rival picks.

    `rivals.tilt_inputs` is now on the weekly run's main path, and it reads one
    endpoint per rival - 45 of them. Left unguarded, `test_pipeline.py` alone
    would make several hundred live requests per run, which is slow, flaky and
    rude to a public API.

    Raising rather than returning empty is deliberate: field_ownership_or_none
    catches this and falls back to the template, so every test exercises the
    fallback path, and a test that genuinely wants field ownership has to say
    so by patching over this.
    """
    import rivals

    def _blocked(url):
        raise RuntimeError(f"the test suite must not fetch {url}")

    monkeypatch.setattr(rivals.PublicFPL, "_fetch", staticmethod(_blocked))
