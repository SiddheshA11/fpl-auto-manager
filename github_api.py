"""
The GitHub REST calls this repo makes, and no more.

Split out because two callers need it and they must not depend on each other:
`manager.py` writes the submission marker after it submits, and `watchdog.py`
reads that marker to decide whether anything actually happened. A watchdog that
imported the thing it is watching would fail with it.

stdlib only, for the same reason - the watchdog has to run when the dependency
install is what broke.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger("fpl_auto")

# Written by manager.py after a real submission, read by the watchdog and by
# the weekly run's gate. It is what makes an hourly schedule idempotent and
# what lets the watchdog watch the outcome instead of the cron.
MARKER_VARIABLE = "FPL_LAST_SUBMITTED_GW"
HEARTBEAT_VARIABLE = "FPL_WATCHDOG_HEARTBEAT"


class GitHubRepo:
    """
    The GitHub REST calls this needs, and no more. `opener` is the test seam;
    nothing here is exercised against the live API by the suite.
    """

    def __init__(self, repo: str, token: str, opener=None):
        self.repo = repo
        self._token = token
        self._opener = opener or self._urlopen

    def _urlopen(self, url: str, method: str, payload: dict | None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
                return resp.status, (json.loads(body) if body.strip() else {})
        except urllib.error.HTTPError as e:
            return e.code, {"message": e.read().decode("utf-8", "replace")[:300]}
        except Exception as e:  # noqa: BLE001 - the watchdog may not crash
            logger.warning("github call failed: %s: %s", type(e).__name__, e)
            return None, {}

    def _api(self, path: str, method: str = "GET", payload: dict | None = None):
        return self._opener(f"https://api.github.com/repos/{self.repo}{path}", method, payload)

    def get_variable(self, name: str) -> str | None:
        status, body = self._api(f"/actions/variables/{name}")
        if status == 200:
            return body.get("value")
        if status == 404:
            return None
        logger.warning("could not read variable %s: status %s", name, status)
        return None

    def set_variable(self, name: str, value: str) -> bool:
        # PATCH first: the variable usually exists, so this is one call in the
        # common case and two only when creating it.
        status, _ = self._api(f"/actions/variables/{name}", "PATCH",
                              {"name": name, "value": value})
        if status in (200, 204):
            return True
        if status == 404:
            status, _ = self._api("/actions/variables", "POST",
                                  {"name": name, "value": value})
            if status in (200, 201, 204):
                return True
        logger.warning("could not write variable %s: status %s", name, status)
        return False

    def dispatch_workflow(self, workflow: str, ref: str = "main", inputs: dict | None = None) -> bool:
        status, body = self._api(f"/actions/workflows/{workflow}/dispatches", "POST",
                                 {"ref": ref, "inputs": inputs or {}})
        if status in (200, 204):
            return True
        logger.error("could not dispatch %s: status %s %s", workflow, status, body.get("message", ""))
        return False
