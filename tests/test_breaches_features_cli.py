import json

import pytest

from pwanalysis import cli
from pwanalysis.breaches import analyse, password_storage
from pwanalysis.config import load_config
from pwanalysis.estimator import GuessEstimator
from pwanalysis.features import featurize, mask

CFG = load_config()


def _b(name, year, pwn, classes, desc="", **flags):
    return {"Name": name, "Title": name, "BreachDate": f"{year}-01-01", "AddedDate": f"{year}-07-01T00:00:00Z",
            "PwnCount": pwn, "DataClasses": classes, "Description": desc, "IsVerified": True,
            "IsFabricated": flags.get("fab", False), "IsSpamList": False}


def test_password_storage_weakest_wins():
    assert password_storage("passwords stored as <a>salted MD5</a> and some bcrypt") == "md5"
    assert password_storage("plain text passwords") == "plaintext"
    assert password_storage("SHA-1 hashes") == "sha1"
    assert password_storage("bcrypt hashes") == "bcrypt"
    assert password_storage("email addresses") is None


def test_breach_analysis():
    data = [_b("a", 2015, 100, ["Emails", "Passwords"], "unsalted MD5"),
            _b("b", 2016, 50, ["Emails"]),
            _b("c", 2016, 10, ["Passwords"], "bcrypt"),
            _b("fake", 2016, 999, ["Passwords"], fab=True)]
    r = analyse(data)
    assert r["n_breaches"] == 3 and r["n_excluded_fabricated_or_spam"] == 1
    assert r["n_with_passwords"] == 2
    assert r["accounts_with_passwords"] == 110
    assert r["password_storage_counts"] == {"md5": 1, "bcrypt": 1}
    assert r["share_stated_storage_plaintext_or_fast"] == 0.5
    json.dumps(r, default=str)


def test_mask():
    assert mask("Pass1!") == "ulllds"


def test_featurize_no_leakage_columns():
    est = GuessEstimator({"passwords": {"password": 2}, "english": {}}, exclude_exact=True)
    X = featurize(["Password1", "xK9#mQ2$"], est, CFG["policies"])
    assert len(X) == 2
    assert not any("pwned" in c or "wl_rank" in c or "blocklist" in c for c in X.columns)
    assert X.loc[0, "policy_legacy_complexity"] == 1
    assert X.loc[0, "cover_dictionary"] > 0


def test_cli_offline_never_prints_password(capsys, monkeypatch):
    resources = ({"passwords": {"password": 2}, "english": {}}, None, None)
    monkeypatch.setattr(cli, "load_resources", lambda cfg: resources)
    secret = "Sup3rSecretValue!"
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(secret + "\n"))
    assert cli.main(["--offline", "--stdin"]) == 0
    out = capsys.readouterr().out
    assert secret not in out and "Time to crack" in out


def test_cli_online_breached_uses_rank(monkeypatch):
    import numpy as np
    from pwanalysis.rank import RankEstimator
    import pwanalysis.hibp as hibp
    monkeypatch.setattr(hibp.HIBPClient, "pwned_count", lambda self, p: 1000)
    rank = RankEstimator(np.array([1, 1, 2, 5]), 1, np.array([1000]))
    r = cli.score("whatever", CFG, offline=False, resources=({}, None, rank))
    assert r["pwned_count"] == 1000
    assert "empirical" in r["basis"]
    assert any("breaches" in t for t in r["feedback"])


@pytest.mark.network
def test_live_pwned_count_for_password():
    from pwanalysis.hibp import HIBPClient
    assert HIBPClient(CFG).pwned_count("password") > 1_000_000
