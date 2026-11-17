"""Guess-number estimation: naive entropy measures and a pattern-aware estimator.

The pattern-aware estimate finds the cheapest way to build the password from a sequence of matches
(dictionary words, keyboard walks, sequences, repeats, dates) with brute-force filling the gaps,
using the zxcvbn cost model:

    guesses = l! * prod(match guesses) + D^(l - 1),   D = 10,000

where ``l`` is the number of pieces. The ``l!`` term charges for not knowing the order of the
pieces and ``D^(l-1)`` makes longer compositions cost more than any shorter one. Brute-force
pieces cost ``cardinality ** length`` and two brute-force pieces may not be adjacent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .patterns import (Match, char_class_cardinality, date_matches, dictionary_matches,
                       repeat_matches, sequence_matches, spatial_matches)

MIN_GUESSES_BEFORE_GROWING_SEQUENCE = 10_000
MIN_SUBMATCH_GUESSES_SINGLE_CHAR = 10
MIN_SUBMATCH_GUESSES_MULTI_CHAR = 50
MAX_ANALYSED_LENGTH = 64


# --------------------------------------------------------------------------- naive measures
def charset_entropy_bits(password: str) -> float:
    """log2(pool ** length): the 'entropy' most strength meters and policies implicitly assume."""
    if not password:
        return 0.0
    return len(password) * math.log2(char_class_cardinality(password))


def shannon_entropy_bits(password: str) -> float:
    """Empirical Shannon entropy of the character distribution, times length."""
    if not password:
        return 0.0
    n = len(password)
    h = -sum(c / n * math.log2(c / n) for c in Counter(password).values())
    return h * n


# --------------------------------------------------------------------------- dictionaries
def load_ranked_list(path: str | Path, limit: int | None = None, min_len: int = 1) -> dict[str, int]:
    ranked: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            w = line.strip().lower()
            if len(w) >= min_len and w not in ranked:
                ranked[w] = len(ranked) + 1
                if limit and len(ranked) >= limit:
                    break
    return ranked


# --------------------------------------------------------------------------- estimator
@dataclass
class Estimate:
    password_length: int
    log10_guesses: float
    sequence: list[Match] = field(default_factory=list)

    @property
    def guesses(self) -> float:
        return 10 ** self.log10_guesses


class GuessEstimator:
    def __init__(self, dictionaries: dict[str, dict[str, int]] | None = None, exclude_exact: bool = False):
        self.dictionaries = dictionaries or {}
        self.exclude_exact = exclude_exact

    def all_matches(self, password: str) -> list[Match]:
        m = dictionary_matches(password, self.dictionaries, exclude_exact=self.exclude_exact)
        m += spatial_matches(password)
        m += sequence_matches(password)
        m += date_matches(password)
        m += repeat_matches(password, lambda block: self._sub_estimate(block))
        return m

    def _sub_estimate(self, block: str) -> float:
        if len(block) == 1:
            return char_class_cardinality(block)
        sub = GuessEstimator(self.dictionaries, exclude_exact=False)
        return sub.estimate(block).guesses

    def estimate(self, password: str) -> Estimate:
        pw = password[:MAX_ANALYSED_LENGTH]
        n = len(pw)
        if n == 0:
            return Estimate(0, 0.0, [])
        matches = self.all_matches(pw)
        by_end: dict[int, list[Match]] = {}
        for m in matches:
            if m.length < n:  # sub-matches must cost at least a minimum
                floor = MIN_SUBMATCH_GUESSES_SINGLE_CHAR if m.length == 1 else MIN_SUBMATCH_GUESSES_MULTI_CHAR
                m.guesses = max(m.guesses, floor)
            by_end.setdefault(m.j, []).append(m)

        # best[k][l][bf] = (log10 product, backpointer)  for prefix ending at k using l pieces,
        # bf = whether the last piece is brute force
        best: list[dict[tuple[int, bool], tuple[float, tuple | None, Match]]] = [dict() for _ in range(n)]

        def relax(k: int, l: int, bf: bool, logp: float, prev, m: Match) -> None:
            key = (l, bf)
            cur = best[k].get(key)
            if cur is None or logp < cur[0]:
                best[k][key] = (logp, prev, m)

        for k in range(n):
            for m in by_end.get(k, []):
                lg = math.log10(max(m.guesses, 1.0))
                if m.i == 0:
                    relax(k, 1, False, lg, None, m)
                else:
                    for (l, bf), (logp, _, _) in list(best[m.i - 1].items()):
                        relax(k, l + 1, False, logp + lg, (m.i - 1, l, bf), m)
            for i in range(k + 1):  # brute-force piece covering pw[i..k]
                tok = pw[i:k + 1]
                lg = len(tok) * math.log10(char_class_cardinality(tok))
                bfm = Match(i, k, tok, "bruteforce", 10 ** min(lg, 300))
                if i == 0:
                    relax(k, 1, True, lg, None, bfm)
                else:
                    for (l, bf), (logp, _, _) in list(best[i - 1].items()):
                        if not bf:
                            relax(k, l + 1, True, logp + lg, (i - 1, l, bf), bfm)

        best_total, best_key = math.inf, None
        for (l, bf), (logp, _, _) in best[n - 1].items():
            a = math.log10(math.factorial(l)) + logp
            b = (l - 1) * math.log10(MIN_GUESSES_BEFORE_GROWING_SEQUENCE)
            total = max(a, b) + math.log10(1 + 10 ** (min(a, b) - max(a, b))) if l > 1 else a
            if total < best_total:
                best_total, best_key = total, (n - 1, l, bf)

        seq, key = [], best_key
        while key is not None:
            k, l, bf = key
            _, prev, m = best[k][(l, bf)]
            seq.append(m)
            key = prev
        return Estimate(n, best_total, seq[::-1])
