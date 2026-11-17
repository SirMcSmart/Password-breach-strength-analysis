"""Pattern matchers for password guess estimation.

Each matcher returns ``Match`` objects covering ``password[i : j + 1]`` with an estimate of how many
guesses an attacker who models that pattern needs to hit the token. The approach follows the
published zxcvbn model (Wheeler, USENIX Security 2016), reimplemented here in simplified form:

* dictionary: ranked word lists, with case, l33t and reversal variations
* spatial: QWERTY keyboard walks (``qwerty``, ``1qaz2wsx``, ``zxcvbn``)
* sequence: constant-step runs (``abcd``, ``9753``)
* repeat: repeated characters or blocks (``aaaa``, ``abcabc``)
* date / year: ``1987``, ``12/05/1999``, ``19990512``
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

REFERENCE_YEAR = 2026
MIN_YEAR_SPACE = 20

LEET = {"4": "a", "@": "a", "8": "b", "(": "c", "{": "c", "3": "e", "6": "g", "9": "g",
        "1": "i", "!": "i", "|": "i", "0": "o", "$": "s", "5": "s", "7": "t", "+": "t", "2": "z"}
LEET_ALT = {"1": "l", "|": "l", "9": "q", "7": "l"}  # second reading for ambiguous symbols


@dataclass
class Match:
    i: int
    j: int
    token: str
    pattern: str
    guesses: float
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.j - self.i + 1


def n_ck(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def char_class_cardinality(s: str) -> int:
    card = 0
    if any(c.islower() for c in s):
        card += 26
    if any(c.isupper() for c in s):
        card += 26
    if any(c.isdigit() for c in s):
        card += 10
    if any(not c.isalnum() for c in s):
        card += 33
    return max(card, 10)


# --------------------------------------------------------------------------- dictionary
def uppercase_variations(token: str) -> int:
    letters = [c for c in token if c.isalpha()]
    if not letters or all(c.islower() for c in letters):
        return 1
    if (letters[0].isupper() and all(c.islower() for c in letters[1:])) \
            or (letters[-1].isupper() and all(c.islower() for c in letters[:-1])) \
            or all(c.isupper() for c in letters):
        return 2
    u = sum(c.isupper() for c in letters)
    lo = len(letters) - u
    return sum(n_ck(u + lo, k) for k in range(1, min(u, lo) + 1))


def l33t_variations(token: str, subs: dict[str, str]) -> int:
    variations = 1
    low = token.lower()
    for subbed, letter in subs.items():
        s, u = low.count(subbed), low.count(letter)
        if s == 0:
            continue
        if u == 0:
            variations *= 2
        else:
            variations *= sum(n_ck(u + s, k) for k in range(1, min(u, s) + 1))
    return variations


def _unleet_variants(token: str) -> list[tuple[str, dict[str, str]]]:
    """De-l33t a lowercase token. Returns (word, {symbol: letter}) for up to two readings."""
    out = []
    for table in (LEET, {**LEET, **LEET_ALT}):
        subs = {c: table[c] for c in set(token) if c in table}
        if subs:
            out.append(("".join(table.get(c, c) for c in token), subs))
    seen, uniq = set(), []
    for w, s in out:
        if w not in seen:
            seen.add(w)
            uniq.append((w, s))
    return uniq


def dictionary_matches(password: str, dictionaries: dict[str, dict[str, int]],
                       min_len: int = 3, exclude_exact: bool = False) -> list[Match]:
    """Substring lookups in ranked dictionaries (+ l33t and reversed variants).

    ``exclude_exact`` skips a match covering the whole password in the ``passwords`` dictionary, so
    a password is never scored by looking itself up (see train.py on why this matters).
    """
    matches: list[Match] = []
    n = len(password)
    low = password.lower()
    for name, ranked in dictionaries.items():
        for i in range(n):
            for j in range(i + min_len - 1, n):
                if exclude_exact and name == "passwords" and i == 0 and j == n - 1:
                    continue
                tok, tok_low = password[i:j + 1], low[i:j + 1]
                rank = ranked.get(tok_low)
                if rank is not None:
                    matches.append(Match(i, j, tok, "dictionary", rank * uppercase_variations(tok),
                                         {"dict": name, "rank": rank, "word": tok_low}))
                rev = ranked.get(tok_low[::-1])
                if rev is not None and tok_low != tok_low[::-1]:
                    matches.append(Match(i, j, tok, "dictionary", rev * uppercase_variations(tok) * 2,
                                         {"dict": name, "rank": rev, "word": tok_low[::-1], "reversed": True}))
                if any(c in LEET for c in tok_low):
                    for word, subs in _unleet_variants(tok_low):
                        r = ranked.get(word)
                        if r is not None and word != tok_low:
                            g = r * uppercase_variations(tok) * l33t_variations(tok, subs)
                            matches.append(Match(i, j, tok, "dictionary", g,
                                                 {"dict": name, "rank": r, "word": word, "l33t": True}))
    return matches


# --------------------------------------------------------------------------- keyboard walks
_ROWS = ["`1234567890-=", " qwertyuiop[]\\", " asdfghjkl;'", " zxcvbnm,./"]
_SHIFTED = {'~': '`', '!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7', '*': '8',
            '(': '9', ')': '0', '_': '-', '+': '=', '{': '[', '}': ']', '|': '\\', ':': ';', '"': "'",
            '<': ',', '>': '.', '?': '/'}
_POS = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row) if ch != " "}
# slanted keyboard: 6 neighbours per key (left, right, up, up-right, down-left, down)
_DIRS = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)]
_KEYS = len(_POS)


def _unshift(c: str) -> tuple[str, bool]:
    if c in _SHIFTED:
        return _SHIFTED[c], True
    if c.isalpha() and c.isupper():
        return c.lower(), True
    return c, False


def _direction(a: str, b: str) -> int | None:
    pa, pb = _POS.get(a), _POS.get(b)
    if pa is None or pb is None:
        return None
    d = (pb[0] - pa[0], pb[1] - pa[1])
    return _DIRS.index(d) if d in _DIRS else None


def _avg_degree() -> float:
    deg = [sum(((r + dr, c + dc) in _POS.values()) for dr, dc in _DIRS) for r, c in _POS.values()]
    return sum(deg) / len(deg)


_AVG_DEGREE = _avg_degree()


def spatial_guesses(length: int, turns: int, shifted: int) -> float:
    g = 0.0
    for i in range(2, length + 1):
        for j in range(1, min(turns, i - 1) + 1):
            g += n_ck(i - 1, j - 1) * _KEYS * _AVG_DEGREE ** j
    if shifted:
        unshifted = length - shifted
        g *= 2 if unshifted == 0 else sum(n_ck(shifted + unshifted, k) for k in range(1, min(shifted, unshifted) + 1))
    return g


def spatial_matches(password: str, min_len: int = 3) -> list[Match]:
    matches = []
    n, i = len(password), 0
    while i < n - 1:
        j, turns, last_dir, shifted = i, 0, None, int(_unshift(password[i])[1])
        while j + 1 < n:
            a, _ = _unshift(password[j])
            b, sh = _unshift(password[j + 1])
            d = _direction(a, b)
            if d is None:
                break
            if d != last_dir:
                turns += 1
                last_dir = d
            shifted += sh
            j += 1
        if j - i + 1 >= min_len:
            tok = password[i:j + 1]
            matches.append(Match(i, j, tok, "spatial", spatial_guesses(len(tok), turns, shifted),
                                 {"turns": turns, "shifted": shifted}))
        i = j if j > i else i + 1
    return matches


# --------------------------------------------------------------------------- sequences
def sequence_matches(password: str, min_len: int = 3) -> list[Match]:
    matches = []
    n = len(password)

    def emit(i: int, j: int, delta: int) -> None:
        if j - i + 1 < min_len or abs(delta) not in (1, 2):
            return
        tok = password[i:j + 1]
        first = tok[0]
        base = 4 if first in "aAzZ019" else (10 if first.isdigit() else 26)
        g = base * len(tok) * (2 if delta < 0 else 1) * (2 if abs(delta) == 2 else 1)
        matches.append(Match(i, j, tok, "sequence", g, {"delta": delta}))

    def same_class(a: str, b: str) -> bool:
        return (a.islower() and b.islower()) or (a.isupper() and b.isupper()) or (a.isdigit() and b.isdigit())

    i = 0
    while i < n - 1:
        delta = ord(password[i + 1]) - ord(password[i])
        if not same_class(password[i], password[i + 1]) or delta == 0:
            i += 1
            continue
        j = i + 1
        while j + 1 < n and same_class(password[j], password[j + 1]) and ord(password[j + 1]) - ord(password[j]) == delta:
            j += 1
        emit(i, j, delta)
        i = j
    return matches


# --------------------------------------------------------------------------- repeats
def repeat_matches(password: str, base_guesses) -> list[Match]:
    """``base_guesses(block)`` estimates the repeated block itself (recursion into the estimator)."""
    matches = []
    for m in re.finditer(r"(?=((.+?)\2+))", password):
        whole, block = m.group(1), m.group(2)
        if len(whole) < 3:
            continue
        i = m.start()
        # prefer the shortest repeating unit of the greedy run
        g = base_guesses(block) * (len(whole) // len(block))
        matches.append(Match(i, i + len(whole) - 1, whole, "repeat", g, {"block": block, "count": len(whole) // len(block)}))
    return matches


# --------------------------------------------------------------------------- dates
_YEAR_RE = re.compile(r"(?<!\d)(19\d\d|20[0-3]\d)(?!\d)")
_DATE_SEP_RE = re.compile(r"(?<!\d)(\d{1,4})([\s/\\_.\-])(\d{1,2})\2(\d{1,4})(?!\d)")


def _year_space(year: int) -> int:
    return max(abs(year - REFERENCE_YEAR), MIN_YEAR_SPACE)


def _valid_dmy(a: int, b: int, c: int) -> int | None:
    """Return a 4-digit year if (a, b, c) reads as a valid date with the year last or first."""
    for y, p, q in ((c, a, b), (a, b, c)):
        if not (y >= 1000 or y <= 99):
            continue
        yy = y if y >= 1000 else y + (1900 if y > 50 else 2000)
        if 1900 <= yy <= 2039 and ((1 <= p <= 12 and 1 <= q <= 31) or (1 <= q <= 12 and 1 <= p <= 31)):
            return yy
    return None


def date_matches(password: str) -> list[Match]:
    matches = []
    for m in _YEAR_RE.finditer(password):
        y = int(m.group(1))
        matches.append(Match(m.start(), m.end() - 1, m.group(1), "date", _year_space(y), {"year": y}))
    for m in _DATE_SEP_RE.finditer(password):
        y = _valid_dmy(int(m.group(1)), int(m.group(3)), int(m.group(4)))
        if y:
            matches.append(Match(m.start(), m.end() - 1, m.group(0), "date", 365 * _year_space(y) * 4,
                                 {"year": y, "separator": m.group(2)}))
    # separator-less dates of 6 or 8 digits: ddmmyy, mmddyyyy, yyyymmdd ...
    for size in (8, 6):
        for i in range(len(password) - size + 1):
            s = password[i:i + size]
            if not s.isdigit() or (i > 0 and password[i - 1].isdigit()) or \
                    (i + size < len(password) and password[i + size].isdigit()):
                continue
            splits = [(s[:2], s[2:4], s[4:])] if size == 6 else [(s[:2], s[2:4], s[4:]), (s[:4], s[4:6], s[6:])]
            for a, b, c in splits:
                y = _valid_dmy(int(a), int(b), int(c))
                if y:
                    matches.append(Match(i, i + size - 1, s, "date", 365 * _year_space(y), {"year": y}))
                    break
    return matches
