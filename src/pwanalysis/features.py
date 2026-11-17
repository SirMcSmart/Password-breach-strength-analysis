"""Feature extraction for the crack-time model.

Every feature is computable offline from the password alone plus the attacker's dictionaries.
Two things are deliberately NOT features:

* the password's rank in the leaked wordlist and its own breach count: they are the same
  frequency signal as the target, so using them would just look the answer up;
* the blocklist part of policy compliance, for the same reason.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .estimator import GuessEstimator, charset_entropy_bits, shannon_entropy_bits
from .policy import char_classes, composition_compliant

PATTERNS = ["dictionary", "spatial", "sequence", "repeat", "date", "bruteforce"]
_YEAR = re.compile(r"(19\d\d|20[0-3]\d)")


def mask(password: str) -> str:
    """hashcat-style mask: u=upper, l=lower, d=digit, s=symbol."""
    return "".join("u" if c.isupper() else "l" if c.islower() else "d" if c.isdigit() else "s" for c in password)


def structure_features(pw: str) -> dict[str, float]:
    n = max(len(pw), 1)
    cls = char_classes(pw)
    m = mask(pw)
    trailing_digits = len(m) - len(m.rstrip("d"))
    runs = [len(r) for r in re.findall(r"(.)\1*", pw)] or [0]
    return {
        "length": len(pw),
        "n_lower": sum(c.islower() for c in pw),
        "n_upper": sum(c.isupper() for c in pw),
        "n_digit": sum(c.isdigit() for c in pw),
        "n_symbol": sum(not c.isalnum() for c in pw),
        "n_classes": sum(cls.values()),
        "frac_digit": sum(c.isdigit() for c in pw) / n,
        "unique_char_ratio": len(set(pw)) / n,
        "max_run": max(runs),
        "trailing_digits": trailing_digits,
        "trailing_symbol": int(bool(pw) and not pw[-1].isalnum()),
        "capitalized_first": int(len(pw) > 1 and pw[0].isupper() and not any(c.isupper() for c in pw[1:])),
        "all_digits": int(pw.isdigit()),
        "all_lower_alpha": int(pw.isalpha() and pw.islower()),
        "has_year": int(bool(_YEAR.search(pw))),
        "n_mask_segments": len(re.findall(r"(.)\1*", m)),
        "charset_entropy_bits": charset_entropy_bits(pw),
        "shannon_entropy_bits": shannon_entropy_bits(pw),
        "shannon_bits_per_char": shannon_entropy_bits(pw) / n,
    }


def pattern_features(pw: str, est: GuessEstimator) -> dict[str, float]:
    r = est.estimate(pw)
    n = max(len(pw), 1)
    cover = {p: 0 for p in PATTERNS}
    best_rank = {"passwords": math.inf, "english": math.inf}
    l33t = reversed_ = 0
    for m in r.sequence:
        cover[m.pattern] += m.length
        if m.pattern == "dictionary":
            d = m.meta["dict"]
            best_rank[d] = min(best_rank.get(d, math.inf), m.meta["rank"])
            l33t |= int(bool(m.meta.get("l33t")))
            reversed_ |= int(bool(m.meta.get("reversed")))
    feats = {f"cover_{p}": cover[p] / n for p in PATTERNS}
    for d, rk in best_rank.items():
        feats[f"log10_best_rank_{d}"] = math.log10(rk) if rk != math.inf else 7.0  # 7 = "no match"
    feats.update({
        "pattern_log10_guesses": r.log10_guesses,
        "n_pieces": len(r.sequence),
        "uses_l33t": l33t,
        "uses_reversed": reversed_,
        "longest_piece_frac": max((m.length for m in r.sequence), default=0) / n,
    })
    return feats


def policy_features(pw: str, policies: dict[str, dict]) -> dict[str, float]:
    return {f"policy_{k}": int(composition_compliant(pw, v)) for k, v in policies.items()}


def featurize(passwords: Iterable[str], est: GuessEstimator, policies: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for pw in passwords:
        f = structure_features(pw)
        f.update(pattern_features(pw, est))
        f.update(policy_features(pw, policies))
        rows.append(f)
    return pd.DataFrame(rows).astype(np.float64)
