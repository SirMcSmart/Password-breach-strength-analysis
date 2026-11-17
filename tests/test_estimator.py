import math

import pytest

from pwanalysis.estimator import GuessEstimator, charset_entropy_bits, shannon_entropy_bits
from pwanalysis.patterns import (date_matches, l33t_variations, sequence_matches, spatial_matches,
                                 uppercase_variations)

DICTS = {"passwords": {"password": 2, "monkey": 14, "dragon": 9, "qwerty": 4},
         "english": {"correct": 900, "horse": 1200, "battery": 2400, "staple": 5100}}


@pytest.fixture
def est():
    return GuessEstimator(DICTS)


def test_charset_entropy():
    assert charset_entropy_bits("abc") == pytest.approx(3 * math.log2(26))
    assert charset_entropy_bits("aB3!") == pytest.approx(4 * math.log2(95))
    assert charset_entropy_bits("") == 0


def test_shannon_entropy():
    assert shannon_entropy_bits("aaaa") == 0
    assert shannon_entropy_bits("abcd") == pytest.approx(8.0)


def test_uppercase_variations():
    assert uppercase_variations("password") == 1
    assert uppercase_variations("Password") == 2
    assert uppercase_variations("PASSWORD") == 2
    assert uppercase_variations("PaSsword") > 2


def test_l33t_variations():
    assert l33t_variations("p4ssword", {"4": "a"}) == 2


def test_dictionary_case_and_l33t(est):
    for pw in ("password", "Password", "P4ssw0rd", "drowssap"):
        r = est.estimate(pw)
        assert r.sequence[0].pattern == "dictionary", pw
        assert r.log10_guesses < 3, pw


def test_exclude_exact_skips_self_lookup():
    e = GuessEstimator(DICTS, exclude_exact=True)
    assert not any(m.pattern == "dictionary" and m.length == 8 for m in e.estimate("password").sequence)
    # sub-words of a longer password still match
    assert any(m.meta.get("word") == "password" for m in e.estimate("password99").sequence)


def test_spatial_walks():
    tokens = {m.token for m in spatial_matches("xqwertyx")}
    assert "qwerty" in tokens
    assert any(m.token == "1qaz" for m in spatial_matches("1qaz"))
    assert spatial_matches("qp") == []


def test_sequences():
    toks = {m.token for m in sequence_matches("abcdef")}
    assert "abcdef" in toks
    assert any(m.meta["delta"] == -1 for m in sequence_matches("9876"))


def test_dates():
    years = {m.meta["year"] for m in date_matches("born1987")}
    assert 1987 in years
    assert any(m.token == "12/05/1999" for m in date_matches("12/05/1999"))
    assert any(m.token == "19871231" for m in date_matches("19871231"))
    assert date_matches("123456") == [] or all(m.token != "123456" for m in date_matches("123456"))


def test_patterns_far_weaker_than_charset_entropy(est):
    for pw in ("Password1987", "qwerty123", "aaaaaaaaaa", "abcdefgh"):
        r = est.estimate(pw)
        naive = charset_entropy_bits(pw) * math.log10(2)
        assert r.log10_guesses < naive - 3, pw


def test_random_password_is_strong(est):
    assert est.estimate("xK9#mQ2$vL7&").log10_guesses > 20


def test_passphrase_beats_complex_short(est):
    assert est.estimate("correcthorsebatterystaple").log10_guesses > est.estimate("Password1!").log10_guesses


def test_monotone_in_length_for_random_suffix(est):
    assert est.estimate("monkeyx7Kq").log10_guesses > est.estimate("monkey").log10_guesses


def test_empty(est):
    assert est.estimate("").log10_guesses == 0
