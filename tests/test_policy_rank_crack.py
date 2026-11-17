import math

import numpy as np
import pytest

from pwanalysis.config import load_config
from pwanalysis.crack_time import crack_seconds, crack_table, humanize
from pwanalysis.hibp import parse_range_body, sha1_hex
from pwanalysis.policy import check_all, check_policy, composition_compliant
from pwanalysis.rank import N_BUCKETS, RankEstimator

CFG = load_config()
POL = CFG["policies"]


def test_legacy_complexity_admits_password1():
    assert composition_compliant("Password1", POL["legacy_complexity"])
    assert not composition_compliant("password", POL["legacy_complexity"])


def test_nist_length_and_blocklist():
    r = check_policy("correct horse battery staple", "n", POL["nist_800_63b_rev4"], pwned_count=0)
    assert r.compliant
    r = check_policy("correct horse battery staple", "n", POL["nist_800_63b_rev4"], pwned_count=5)
    assert not r.compliant and "breach" in r.failures[0]
    assert not check_policy("short", "n", POL["nist_800_63b_rev4"]).compliant


def test_blocklist_skipped_when_unknown():
    r = check_policy("x" * 20, "n", POL["nist_800_63b_rev4"], pwned_count=None)
    assert r.compliant


def test_pci_requires_alpha_and_digit():
    assert not composition_compliant("abcdefghijkl", POL["pci_dss_v4"])
    assert composition_compliant("abcdefghijk1", POL["pci_dss_v4"])


def test_check_all_covers_every_policy():
    assert len(check_all("Password1", POL)) == len(POL)


def test_sha1_and_padding_parse():
    assert sha1_hex("password") == "5BAA61E4C9B93F3F0682250B6CF8331B7EE68FD8"
    body = "1E4C9B93F3F0682250B6CF8331B7EE68FD8:9545824\r\n0000000000000000000000000000000000A:0\r\n"
    assert parse_range_body(body) == {"1E4C9B93F3F0682250B6CF8331B7EE68FD8": 9545824}


def test_rank_estimator_scaling_and_head():
    sample = np.array([1] * 900 + [2] * 90 + [10] * 10)  # 1000 hashes from 2 buckets
    est = RankEstimator(sample, 2, known_counts=np.array([10_000_000, 5_000, 10, 2]))
    assert est.corpus_size == pytest.approx(1000 * N_BUCKETS / 2)
    r = est.rank([10_000_000, 5_000, 10, 2, 1])
    assert r[0] == 1  # top password is the first guess
    assert r[1] == 2
    assert np.all(np.diff(r) > 0)  # rarer -> later
    assert r[-1] == pytest.approx(N_BUCKETS / 2 * (100 + 450))
    with pytest.raises(ValueError):
        est.rank([0])


def test_crack_time():
    assert crack_seconds(6, 1e3) == pytest.approx(1e3)
    assert humanize(0.2).startswith("instant")
    assert humanize(7200) == "2 hours"
    assert humanize(1e20) == "centuries+"
    t = crack_table(3, CFG["attack_scenarios"])
    assert set(t) == set(CFG["attack_scenarios"])
