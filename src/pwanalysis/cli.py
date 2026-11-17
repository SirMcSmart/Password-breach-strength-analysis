"""``pwcheck``: score one password.

The password is read with getpass (or one line from stdin with ``--stdin``), never from argv, so
it does not land in shell history or the process list. It is never printed or logged. The online
breach check sends only the first 5 hex characters of its SHA-1 hash (k-anonymity), with response
padding on.

How the final estimate is chosen:
  * found in Pwned Passwords   -> empirical guess rank from the breach corpus (rank.py)
  * not found (online check)   -> the attacker has to exhaust the corpus first, so the estimate is
                                  max(corpus size, pattern estimate, model estimate)
  * --offline                  -> model estimate; if the pattern estimate is beyond the corpus
                                  size the model was trained on, the larger of the two
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import sys

import joblib
import numpy as np

from .config import load_config, project_path
from .crack_time import crack_table
from .estimator import GuessEstimator, charset_entropy_bits, load_ranked_list, shannon_entropy_bits
from .features import featurize
from .policy import check_all
from .rank import RankEstimator


def load_resources(cfg: dict):
    raw = project_path(cfg["data"]["raw_dir"])
    models = project_path(cfg["artifacts"]["models_dir"])
    dicts, wl = {}, raw / cfg["data"]["wordlist_name"]
    if wl.exists():
        dicts["passwords"] = load_ranked_list(wl)
    eng = raw / cfg["data"]["english_name"]
    if eng.exists():
        dicts["english"] = load_ranked_list(eng, min_len=3)
    model = joblib.load(models / "crack_model.joblib") if (models / "crack_model.joblib").exists() else None
    rank_est = None
    if (models / "rank_estimator.npz").exists():
        z = np.load(models / "rank_estimator.npz")
        rank_est = RankEstimator(z["sample_counts"], int(z["n_sampled_buckets"]), z["known_counts"])
    return dicts, model, rank_est


def feedback(est_seq, policies, pwned: int | None, length: int) -> list[str]:
    tips = []
    if pwned:
        tips.append(f"This password has appeared in breaches {pwned:,} times. Never use it; attackers try these first.")
    kinds = {m.pattern for m in est_seq}
    if "dictionary" in kinds:
        tips.append("Contains common words or previously leaked passwords; capitals, l33t swaps and reversal add little.")
    if kinds & {"spatial", "sequence", "repeat"}:
        tips.append("Contains a keyboard walk, sequence or repetition, which guessing tools enumerate early.")
    if "date" in kinds:
        tips.append("Contains a date or year; dates are a small, well-known search space.")
    if length < 15:
        tips.append("Length matters most: a passphrase of 4+ random, unrelated words (15+ characters) beats symbol rules.")
    if not tips:
        tips.append("No common patterns found. Store it in a password manager and enable MFA where offered.")
    return tips


def score(password: str, cfg: dict, offline: bool = False, resources=None) -> dict:
    dicts, model, rank_est = resources or load_resources(cfg)
    est = GuessEstimator(dicts, exclude_exact=True)
    pat = est.estimate(password)
    exact_rank = dicts.get("passwords", {}).get(password.lower())

    pwned = None
    if not offline:
        from .hibp import HIBPClient
        pwned = HIBPClient(cfg).pwned_count(password)

    model_pred = None
    if model is not None:
        X = featurize([password], est, cfg["policies"])[model["features"]]
        model_pred = float(model["model"].predict(X)[0])
    corpus_log10 = math.log10(rank_est.corpus_size) if rank_est else 9.0

    if pwned and rank_est is not None:
        final, basis = float(np.log10(rank_est.rank([pwned])[0])), "empirical guess rank in the HIBP breach corpus"
    elif pwned == 0:
        final = max(x for x in (corpus_log10, pat.log10_guesses, model_pred) if x is not None)
        basis = "not in the breach corpus: max(corpus size, pattern estimate, model estimate)"
    else:
        if model_pred is None:
            final, basis = pat.log10_guesses, "pattern estimate (no trained model found)"
        elif pat.log10_guesses > corpus_log10:
            final, basis = max(model_pred, pat.log10_guesses), "offline: max(model, pattern estimate)"
        else:
            final, basis = model_pred, "offline: crack-time model"

    return {
        "length": len(password),
        "charset_entropy_bits": round(charset_entropy_bits(password), 1),
        "shannon_entropy_bits": round(shannon_entropy_bits(password), 1),
        "pattern_log10_guesses": round(pat.log10_guesses, 2),
        "model_log10_guesses": None if model_pred is None else round(model_pred, 2),
        "pwned_count": pwned,
        "in_leaked_wordlist_rank": exact_rank,
        "final_log10_guesses": round(final, 2),
        "basis": basis,
        "patterns": [{"pattern": m.pattern, "length": m.length,
                      **{k: v for k, v in m.meta.items() if k in ("dict", "rank", "l33t", "reversed", "year")}}
                     for m in pat.sequence],
        "crack_times": {k: v["display"] for k, v in crack_table(final, cfg["attack_scenarios"]).items()},
        "policies": {r.label: ("pass" if r.compliant else "fail: " + "; ".join(r.failures))
                     for r in check_all(password, cfg["policies"], pwned)},
        "feedback": feedback(pat.sequence, cfg["policies"], pwned, len(password)),
    }


def render(r: dict, cfg: dict) -> str:
    out = [f"Length {r['length']}  |  charset entropy {r['charset_entropy_bits']} bits  |  "
           f"Shannon {r['shannon_entropy_bits']} bits"]
    if r["pwned_count"] is None:
        out.append("Breach check: skipped (--offline)")
    else:
        out.append(f"Breach check: {'FOUND ' + format(r['pwned_count'], ',') + ' times' if r['pwned_count'] else 'not found'} in Pwned Passwords")
    if r["in_leaked_wordlist_rank"]:
        out.append(f"Leaked wordlist: exact match at rank {r['in_leaked_wordlist_rank']:,}")
    out.append("Structure: " + " + ".join(
        p["pattern"] + (f"({p['dict']})" if "dict" in p else "") + f"[{p['length']}]" for p in r["patterns"]))
    out.append(f"Estimated guesses: 10^{r['final_log10_guesses']}  ({r['basis']})")
    out.append(f"  pattern estimator 10^{r['pattern_log10_guesses']}, model "
               f"{'n/a' if r['model_log10_guesses'] is None else '10^' + str(r['model_log10_guesses'])}")
    out.append("\nTime to crack:")
    for k, disp in r["crack_times"].items():
        out.append(f"  {cfg['attack_scenarios'][k]['label']:<40} {disp}")
    out.append("\nPolicy compliance:")
    for lab, res in r["policies"].items():
        out.append(f"  {lab:<40} {res}")
    out.append("\nAdvice:")
    out += [f"  - {t}" for t in r["feedback"]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pwcheck", description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="skip the k-anonymity breach lookup")
    ap.add_argument("--stdin", action="store_true", help="read the password from the first line of stdin")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    cfg = load_config()
    password = sys.stdin.readline().rstrip("\r\n") if args.stdin else getpass.getpass("Password (not echoed): ")
    if not password:
        print("empty password", file=sys.stderr)
        return 2
    r = score(password, cfg, offline=args.offline)
    print(json.dumps(r, indent=1) if args.json else render(r, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
