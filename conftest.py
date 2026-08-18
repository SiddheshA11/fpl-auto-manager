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
