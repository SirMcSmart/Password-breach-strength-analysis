"""Convert guess counts into crack times for the attack scenarios in config.yaml."""

from __future__ import annotations

import math

_UNITS = [("century", 100 * 365.25 * 86400), ("year", 365.25 * 86400), ("month", 30.44 * 86400),
          ("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]


def crack_seconds(log10_guesses: float, rate: float) -> float:
    """Expected seconds to reach the guess number at ``rate`` guesses/second."""
    return 10 ** (log10_guesses - math.log10(rate))


def humanize(seconds: float) -> str:
    if seconds < 1:
        return "instant (< 1 second)"
    if seconds >= 1e4 * _UNITS[0][1]:
        return "centuries+"
    for name, size in _UNITS:
        if seconds >= size:
            n = round(seconds / size)
            unit = name if n == 1 else ("centuries" if name == "century" else name + "s")
            return f"{n:,} {unit}"
    return "instant (< 1 second)"


def crack_table(log10_guesses: float, scenarios: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for key, sc in scenarios.items():
        s = crack_seconds(log10_guesses, float(sc["rate"]))
        out[key] = {"label": sc["label"], "seconds": s, "display": humanize(s)}
    return out
