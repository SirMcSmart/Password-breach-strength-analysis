"""Turn a Pwned Passwords breach count into an attacker guess rank.

An attacker with the breach corpus tries passwords in descending order of how often they were seen.
A password's guess number under that "optimal dictionary attack" is the number of distinct
passwords seen at least as often as it.

The full corpus is ~1 billion hashes, so we estimate that number from ``k`` uniformly random
k-anonymity buckets (out of 16^5): every hash lands in a bucket by its SHA-1 prefix, so the counts
inside random buckets are an unbiased sample of the corpus and the tail count scales by 16^5/k.
The sample cannot resolve the extreme head (a count of 10^7 is never in a random bucket), so the
exact ordering of the looked-up passwords themselves is used as a lower bound there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BUCKETS = 16**5


@dataclass
class RankEstimator:
    sample_counts: np.ndarray  # counts from all hashes in the random buckets
    n_sampled_buckets: int
    known_counts: np.ndarray  # counts of all looked-up passwords (exact, for the head)

    def __post_init__(self) -> None:
        self._sample_asc = np.sort(np.asarray(self.sample_counts, dtype=np.int64))
        self._known_asc = np.sort(np.asarray(self.known_counts, dtype=np.int64))
        self.scale = N_BUCKETS / self.n_sampled_buckets

    @property
    def corpus_size(self) -> float:
        """Estimated number of distinct passwords in Pwned Passwords."""
        return self.scale * len(self._sample_asc)

    def _n_ge(self, arr_asc: np.ndarray, c: np.ndarray) -> np.ndarray:
        return len(arr_asc) - np.searchsorted(arr_asc, c, side="left")

    def rank(self, counts) -> np.ndarray:
        """Estimated guess rank (1 = first guess) for breach counts > 0. Mid-rank within ties."""
        c = np.atleast_1d(np.asarray(counts, dtype=np.int64))
        if np.any(c <= 0):
            raise ValueError("rank is defined for breached passwords only (count > 0)")
        s_hi, s_eq = self._n_ge(self._sample_asc, c + 1), self._n_ge(self._sample_asc, c)
        sample_rank = self.scale * (s_hi + 0.5 * (s_eq - s_hi))
        k_hi = self._n_ge(self._known_asc, c + 1)
        known_rank = k_hi + 1.0
        return np.maximum(np.maximum(sample_rank, known_rank), 1.0)
