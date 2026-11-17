"""HaveIBeenPwned clients.

Pwned Passwords is queried with the k-anonymity range API: only the first five hex characters of
the password's SHA-1 hash leave the machine, and the match is done locally on the returned
suffixes. With ``Add-Padding`` the service pads every response with fake zero-count suffixes, so
even the response size says nothing about which hash was asked for. Padding rows are dropped.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import requests

from .config import load_config


def sha1_hex(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8")).hexdigest().upper()


def parse_range_body(body: str) -> dict[str, int]:
    """Parse ``SUFFIX:COUNT`` lines, dropping padding rows (count 0)."""
    out: dict[str, int] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        suffix, count = line.split(":", 1)
        n = int(count)
        if n > 0:
            out[suffix.upper()] = n
    return out


class HIBPClient:
    def __init__(self, cfg: dict[str, Any] | None = None, session: requests.Session | None = None):
        self.cfg = (cfg or load_config())["hibp"]
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.cfg["user_agent"]})

    def _get(self, url: str, headers: dict[str, str] | None = None) -> requests.Response:
        delay = 1.0
        for attempt in range(self.cfg["max_retries"] + 1):
            try:
                r = self.session.get(url, headers=headers, timeout=self.cfg["timeout_seconds"])
            except requests.RequestException:
                if attempt == self.cfg["max_retries"]:
                    raise
            else:
                if r.status_code == 200:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504) or attempt == self.cfg["max_retries"]:
                    r.raise_for_status()
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    def fetch_range(self, prefix: str) -> dict[str, int]:
        prefix = prefix.upper()
        if len(prefix) != 5 or any(c not in "0123456789ABCDEF" for c in prefix):
            raise ValueError("prefix must be 5 hex characters")
        headers = {"Add-Padding": "true"} if self.cfg.get("add_padding") else None
        r = self._get(self.cfg["range_url"].format(prefix=prefix), headers=headers)
        return parse_range_body(r.text)

    def pwned_count(self, password: str) -> int:
        """Times the password appears in Pwned Passwords (0 = not found). Sends 5 hash chars only."""
        h = sha1_hex(password)
        return self.fetch_range(h[:5]).get(h[5:], 0)

    def fetch_breaches(self) -> list[dict[str, Any]]:
        """All breaches in HIBP (public metadata endpoint, no API key needed)."""
        return self._get(self.cfg["breaches_url"]).json()
