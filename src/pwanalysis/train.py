"""Fit and evaluate the crack-time model.

Target: log10 of the empirical guess rank (rank.py) of each breached password, i.e. how many
guesses an attacker replaying the HIBP corpus in frequency order needs.

Leakage control: the attacker dictionary is the full 1M wordlist *minus the test passwords*, and the
estimator never matches a password against its own exact dictionary entry (``exclude_exact``). So
training and test passwords are featurised the same way, and the model has to generalise from
structure, patterns and related leaked words, not look passwords up.

Compared models (all predict log10 guesses):
  * charset_entropy_raw / shannon_entropy_raw: the "bits of entropy" a typical meter shows,
    converted to guesses (2^bits / 2), uncalibrated
  * *_calibrated: the same single feature, linearly calibrated on the training split
  * pattern_estimator_raw / _calibrated: the zxcvbn-style estimator in estimator.py
  * gbm_structure: gradient boosting on entropy + composition + policy features only
  * gbm_full: gradient boosting on all features (adds pattern-match features)

Usage: ``python -m pwanalysis.train``
"""

from __future__ import annotations

import json
import logging
import math

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .config import load_config, project_path
from .estimator import GuessEstimator, load_ranked_list
from .features import featurize
from .rank import RankEstimator

log = logging.getLogger(__name__)

PATTERN_ONLY = {"pattern_log10_guesses", "n_pieces", "uses_l33t", "uses_reversed", "longest_piece_frac"}


def is_pattern_feature(col: str) -> bool:
    return col in PATTERN_ONLY or col.startswith(("cover_", "log10_best_rank_"))


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, RankEstimator]:
    processed = project_path(cfg["data"]["processed_dir"])
    df = pd.read_csv(processed / "passwords.csv", keep_default_na=False, dtype={"password": str, "prefix": str})
    counts = np.load(processed / "corpus_sample_counts.npy")
    breached = df["pwned_count"] > 0
    rank_est = RankEstimator(counts, cfg["sample"]["n_random_buckets"], df.loc[breached, "pwned_count"].to_numpy())
    df["log10_rank"] = np.nan
    df.loc[breached, "log10_rank"] = np.log10(rank_est.rank(df.loc[breached, "pwned_count"].to_numpy()))
    return df, rank_est


def build_estimator(cfg: dict, exclude: set[str] | None = None) -> GuessEstimator:
    raw = project_path(cfg["data"]["raw_dir"])
    pw_dict = load_ranked_list(raw / cfg["data"]["wordlist_name"])
    if exclude:
        for p in exclude:
            pw_dict.pop(p.lower(), None)
    eng = load_ranked_list(raw / cfg["data"]["english_name"], min_len=3)
    return GuessEstimator({"passwords": pw_dict, "english": eng}, exclude_exact=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    return {
        "spearman": float(spearmanr(y, p).statistic),
        "mae_log10": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "within_1_order": float(np.mean(np.abs(err) <= 1)),
        # security-relevant error: the estimate says >= 10x stronger than the password really is
        "overestimate_10x_rate": float(np.mean(err > 1)),
        "median_error_log10": float(np.median(err)),
    }


def bits_to_log10_guesses(bits: np.ndarray) -> np.ndarray:
    return bits * math.log10(2) - math.log10(2)


def gbm(cfg: dict) -> HistGradientBoostingRegressor:
    m = cfg["model"]
    return HistGradientBoostingRegressor(
        learning_rate=m["learning_rate"], max_iter=m["max_iter"], max_leaf_nodes=m["max_leaf_nodes"],
        min_samples_leaf=m["min_samples_leaf"], l2_regularization=m["l2_regularization"],
        early_stopping=m["early_stopping"], validation_fraction=m["validation_fraction"],
        n_iter_no_change=m["n_iter_no_change"], random_state=cfg["seed"])


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    df, rank_est = load_dataset(cfg)
    data = df[df["pwned_count"] > 0].reset_index(drop=True)
    unbreached = df[df["pwned_count"] == 0].reset_index(drop=True)
    tr_idx, te_idx = train_test_split(np.arange(len(data)), test_size=cfg["model"]["test_size"], random_state=cfg["seed"])
    test_pw = set(data.loc[te_idx, "password"])
    log.info("breached: %d (train %d / test %d), not in HIBP: %d", len(data), len(tr_idx), len(te_idx), len(unbreached))

    est = build_estimator(cfg, exclude=test_pw)
    X = featurize(data["password"], est, cfg["policies"])
    X_unb = featurize(unbreached["password"], est, cfg["policies"]) if len(unbreached) else None
    y = data["log10_rank"].to_numpy()
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y[tr_idx], y[te_idx]

    struct_cols = [c for c in X.columns if not is_pattern_feature(c)]
    preds: dict[str, np.ndarray] = {}
    for name, col in [("charset_entropy", "charset_entropy_bits"), ("shannon_entropy", "shannon_entropy_bits")]:
        preds[f"{name}_raw"] = bits_to_log10_guesses(Xte[col].to_numpy())
        lr = LinearRegression().fit(Xtr[[col]], ytr)
        preds[f"{name}_calibrated"] = lr.predict(Xte[[col]])
    preds["pattern_estimator_raw"] = Xte["pattern_log10_guesses"].to_numpy()
    lr = LinearRegression().fit(Xtr[["pattern_log10_guesses"]], ytr)
    preds["pattern_estimator_calibrated"] = lr.predict(Xte[["pattern_log10_guesses"]])

    m_struct = gbm(cfg).fit(Xtr[struct_cols], ytr)
    preds["gbm_structure"] = m_struct.predict(Xte[struct_cols])
    m_full = gbm(cfg).fit(Xtr, ytr)
    preds["gbm_full"] = m_full.predict(Xte)

    results = {k: metrics(yte, v) for k, v in preds.items()}
    for k, v in results.items():
        log.info("%-30s spearman=%.3f mae=%.2f within1=%.2f over10x=%.2f", k, v["spearman"], v["mae_log10"], v["within_1_order"], v["overestimate_10x_rate"])

    pi = permutation_importance(m_full, Xte, yte, n_repeats=5, random_state=cfg["seed"], scoring="neg_mean_absolute_error")
    importance = sorted(zip(X.columns, pi.importances_mean), key=lambda t: -t[1])

    # Censored check: passwords NOT in HIBP must take more guesses than the whole corpus.
    censored = None
    if X_unb is not None and len(X_unb) >= 20:
        floor = math.log10(rank_est.corpus_size)
        p_unb = m_full.predict(X_unb)
        censored = {
            "n": int(len(X_unb)),
            "corpus_log10": floor,
            "gbm_full_median_pred": float(np.median(p_unb)),
            "pattern_estimator_median": float(np.median(X_unb["pattern_log10_guesses"])),
            "breached_test_median_true": float(np.median(yte)),
        }

    # Final served model: refit on all breached passwords.
    final = clone(m_full).fit(X, y)
    models_dir = project_path(cfg["artifacts"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": list(X.columns), "version": 1}, models_dir / "crack_model.joblib")
    np.savez_compressed(models_dir / "rank_estimator.npz", sample_counts=rank_est.sample_counts,
                        n_sampled_buckets=rank_est.n_sampled_buckets, known_counts=rank_est.known_counts)

    processed = project_path(cfg["data"]["processed_dir"])
    test_out = data.loc[te_idx, ["password", "wl_rank", "pwned_count", "log10_rank"]].copy()
    for k, v in preds.items():
        test_out[f"pred_{k}"] = v
    test_out.to_csv(processed / "test_predictions.csv", index=False)
    feats_out = pd.concat([data[["password", "pwned_count", "log10_rank", "source"]], X], axis=1)
    feats_out.to_csv(processed / "features.csv", index=False)

    out = {
        "n_breached": int(len(data)), "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "n_not_in_hibp": int(len(unbreached)),
        "corpus_size_estimate": rank_est.corpus_size,
        "target": "log10(empirical guess rank in HIBP Pwned Passwords)",
        "target_range_test": [float(yte.min()), float(yte.max())],
        "models": results,
        "permutation_importance_top15": [{"feature": f, "mae_increase": float(v)} for f, v in importance[:15]],
        "censored_check": censored,
        "gbm_full_iterations": int(m_full.n_iter_),
    }
    outputs = project_path(cfg["artifacts"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
