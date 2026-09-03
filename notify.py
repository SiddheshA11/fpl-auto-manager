"""
Outbound alerting. Telegram, behind an interface so tests never reach the wire.

`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` have been listed as required
secrets since the repo was set up, and until now no workflow and no module in
the repo referenced either of them. The alerting was designed and never wired,
which is why the GW2 deadline could pass unmentioned.

Two rules govern everything here:

  - a notifier must never take down its caller. The weekly run submitting a
    squad is the valuable thing; failing to announce it is not a reason to
    fail the run. Every path returns a bool and logs, and none of them raise.
  - a missing secret is a loud no-op, not a crash. A fresh clone with no .env
    still runs.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

import requests

logger = logging.getLogger("fpl_auto")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

REQUEST_TIMEOUT = 15
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0

# Telegram rejects anything longer. Truncate rather than lose the message.
MAX_MESSAGE_CHARS = 4096


class Notifier(Protocol):
    def send(self, text: str) -> bool: ...


class NullNotifier:
    """
    Stands in when the secrets are absent. Records what it would have sent so a
    caller can be tested without a transport, and warns once per send so a
    misconfigured deployment is visible in the logs rather than silently mute.
    """

    def __init__(self, reason: str = "no Telegram credentials configured"):
        self.reason = reason
        self.sent: list[str] = []

    def send(self, text: str) -> bool:
        self.sent.append(text)
        logger.warning("alert not delivered (%s): %s", self.reason, text.splitlines()[0][:200])
        return False


class TelegramNotifier:
    """
    `transport` is the seam. It takes (url, json_payload, timeout) and returns
    something with .status_code and .text, which is exactly requests.post - so
    production passes nothing and tests pass a fake.
    """

    def __init__(self, token: str, chat_id: str, transport: Callable | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._token = token
        self._chat_id = chat_id
        self._post = transport or (lambda url, json, timeout: requests.post(url, json=json, timeout=timeout))
        self._sleep = sleep

    def send(self, text: str) -> bool:
        if not text:
            return False
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[: MAX_MESSAGE_CHARS - 3] + "..."

        url = TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        # Retried because a duplicate alert costs nothing and a dropped one
        # costs a gameweek. sendMessage is not idempotent, so this can double a
        # message when a response is lost after delivery; that trade is
        # deliberate and in this direction only.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._post(url, payload, REQUEST_TIMEOUT)
            except Exception as e:  # noqa: BLE001 - a notifier may not raise
                logger.warning("telegram send failed (attempt %d/%d): %s: %s",
                               attempt, MAX_ATTEMPTS, type(e).__name__, e)
            else:
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return True
                # 4xx other than 429 will not improve on a retry: a bad token
                # or a wrong chat_id is a configuration error, not a blip.
                body = str(getattr(resp, "text", ""))[:300]
                if status is not None and 400 <= status < 500 and status != 429:
                    logger.error("telegram rejected the message (%s): %s. "
                                 "Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.", status, body)
                    return False
                logger.warning("telegram send failed (attempt %d/%d): status %s: %s",
                               attempt, MAX_ATTEMPTS, status, body)

            if attempt < MAX_ATTEMPTS:
                self._sleep(BACKOFF_BASE ** (attempt - 1))

        logger.error("telegram send gave up after %d attempts", MAX_ATTEMPTS)
        return False


def from_env(env: dict | None = None) -> Notifier:
    """Build a notifier from the environment, or a NullNotifier explaining why not."""
    env = os.environ if env is None else env
    token = (env.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (env.get("TELEGRAM_CHAT_ID") or "").strip()

    missing = [n for n, v in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)) if not v]
    if missing:
        return NullNotifier(f"missing {', '.join(missing)}")
    return TelegramNotifier(token, chat_id)
