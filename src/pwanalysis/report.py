"""Breach + password-structure analysis, figures, ``outputs/report.md`` and the README results block.

Run after ``pwanalysis.dataset`` and ``pwanalysis.train``. Every number written here is computed
from files in ``data/processed`` and ``outputs/metrics.json``; nothing is typed in by hand.

Usage: ``python -m pwanalysis.report``
"""

from __future__ import annotations

import json
import logging
import math
import re

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .breaches import analyse as analyse_breaches  # noqa: E402
from .config import PROJECT_ROOT, load_config, project_path  # noqa: E402
from .crack_time import humanize  # noqa: E402
from .features import mask  # noqa: E402
from .policy import composition_compliant  # noqa: E402

log = logging.getLogger(__name__)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DAY = 86400

MODEL_LABELS = {
    "charset_entropy_raw": "Charset entropy (raw)",
    "charset_entropy_calibrated": "Charset entropy (calibrated)",
    "shannon_entropy_raw": "Shannon entropy (raw)",
    "shannon_entropy_calibrated": "Shannon entropy (calibrated)",
    "pattern_estimator_raw": "Pattern estimator (raw)",
    "pattern_estimator_calibrated": "Pattern estimator (calibrated)",
    "gbm_structure": "GBM: entropy + composition",
    "gbm_full": "GBM: + pattern features",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "font.size": 10, "figure.dpi": 130, "axes.axisbelow": True,
    })


def _save(fig, name: str, figdir) -> str:
    fig.tight_layout()
    path = figdir / name
    fig.savefig(path)
    plt.close(fig)
    return f"outputs/figures/{name}"


def _hbar(ax, labels, values, fmt="{:.0f}", color=BLUE):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + span * 0.01, yi, fmt.format(v), va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, span * 1.15)


# --------------------------------------------------------------------------- password analysis
def analyse_passwords(cfg: dict, metrics: dict) -> dict:
    processed = project_path(cfg["data"]["processed_dir"])
    pw = pd.read_csv(processed / "passwords.csv", keep_default_na=False, dtype={"password": str, "prefix": str})
    feats = pd.read_csv(processed / "features.csv", keep_default_na=False, dtype={"password": str})
    scen = cfg["attack_scenarios"]

    uniform = pw[pw["in_random_bucket"].astype(str).str.lower() == "true"]
    uni_b = uniform[uniform["pwned_count"] > 0]
    fb = feats.merge(pw[["password", "in_random_bucket"]], on="password", how="left")
    fu = fb[fb["in_random_bucket"].astype(str).str.lower() == "true"]

    def share(s) -> float:
        return float(np.mean(s)) if len(s) else float("nan")

    masks = uni_b["password"].map(mask)
    top_masks = masks.value_counts().head(10)
    patterns = {
        "contains a dictionary/leaked word": share(fu["cover_dictionary"] > 0),
        "capitalised first letter only": share(fu["capitalized_first"] == 1),
        "ends in digits": share(fu["trailing_digits"] > 0),
        "contains a year (1900-2039)": share(fu["has_year"] == 1),
        "all digits": share(fu["all_digits"] == 1),
        "all lowercase letters": share(fu["all_lower_alpha"] == 1),
        "keyboard walk": share(fu["cover_spatial"] > 0),
        "character sequence (abc, 123)": share(fu["cover_sequence"] > 0),
        "repeated characters/blocks": share(fu["cover_repeat"] > 0),
        "l33t substitution": share(fu["uses_l33t"] == 1),
    }

    # crack time by scenario, using the empirical rank of every breached password
    log10_rank = fb["log10_rank"].astype(float).to_numpy()
    uni_rank = fu["log10_rank"].astype(float).to_numpy()  # uniform sample: unbiased shares
    crack_within_day = {}
    for k, sc in scen.items():
        secs = 10 ** (uni_rank - math.log10(float(sc["rate"])))
        crack_within_day[k] = {"label": sc["label"], "share_within_1_day": share(secs <= DAY),
                               "share_within_1_second": share(secs <= 1)}

    policy_rows = []
    for name, rules in cfg["policies"].items():
        comp_uni = uni_b["password"].map(lambda p: composition_compliant(p, rules))
        comp_all = fb["password"].map(lambda p: composition_compliant(p, rules)).to_numpy(bool)
        r = log10_rank[comp_all]
        row = {
            "policy": name, "label": rules["label"], "blocklist": bool(rules.get("blocklist")),
            "share_breached_uniform_pass_composition": share(comp_uni),
            "n_breached_compliant_all": int(comp_all.sum()),
            "median_log10_rank_compliant": float(np.median(r)) if len(r) else None,
            "median_charset_bits_compliant": float(np.median(fb.loc[comp_all, "charset_entropy_bits"])) if len(r) else None,
            "share_compliant_cracked_1day_bcrypt": share(10 ** (r - math.log10(scen["offline_bcrypt_cost10"]["rate"])) <= DAY) if len(r) else None,
        }
        examples = fb.loc[comp_all].sort_values("pwned_count", ascending=False)["password"].head(5).tolist()
        row["most_common_compliant_examples"] = examples
        policy_rows.append(row)

    test = pd.read_csv(processed / "test_predictions.csv", keep_default_na=False, dtype={"password": str})
    gap = (test["pred_charset_entropy_raw"] - test["log10_rank"]).astype(float)
    leg = test["password"].map(lambda p: composition_compliant(p, cfg["policies"]["legacy_complexity"]))
    famous = ["123456", "password", "qwerty", "Password1", "P@ssw0rd", "iloveyou", "letmein", "dragon"]
    fam = pw[pw["password"].isin(famous) & (pw["pwned_count"] > 0)][["password", "pwned_count"]]
    fam = fam.merge(feats[["password", "log10_rank", "charset_entropy_bits"]], on="password", how="left")

    return {
        "uniform_sample_n": int(len(uniform)),
        "uniform_share_in_hibp": share(uniform["pwned_count"] > 0),
        "uniform_breached_n": int(len(uni_b)),
        "length_counts_uniform_breached": uni_b["password"].str.len().clip(upper=20).value_counts().sort_index().to_dict(),
        "median_length_uniform_breached": float(uni_b["password"].str.len().median()),
        "classes_uniform_breached": fu["n_classes"].astype(int).value_counts().sort_index().to_dict(),
        "top_masks_uniform_breached": [{"mask": m, "share": float(c / len(masks))} for m, c in top_masks.items()],
        "pattern_prevalence_uniform_breached": patterns,
        "crack_time_breached": crack_within_day,
        "policies": policy_rows,
        "entropy_overstatement_test_median_log10": float(gap.median()),
        "entropy_overstatement_test_legacy_compliant_median_log10": float(gap[leg].median()) if leg.any() else None,
        "famous_examples": fam.sort_values("pwned_count", ascending=False).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- figures
def make_figures(cfg: dict, br: dict, pa: dict, metrics: dict) -> dict[str, str]:
    _style()
    figdir = project_path(cfg["artifacts"]["outputs_dir"]) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = {}

    py = pd.DataFrame(br["per_year"])
    py = py[py["year"] >= 2005]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(py["year"], py["with_passwords"], color=BLUE, label="Exposed passwords", edgecolor=SURFACE, linewidth=1)
    ax.bar(py["year"], py["breaches"] - py["with_passwords"], bottom=py["with_passwords"], color=ORANGE,
           label="No passwords exposed", edgecolor=SURFACE, linewidth=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("HIBP breaches by year of breach")
    ax.set_ylabel("Breaches")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    figs["breaches_per_year"] = _save(fig, "breaches_per_year.png", figdir)

    sc = br["password_storage_counts"]
    order = ["plaintext", "md5", "sha1", "sha2", "pbkdf2", "bcrypt", "scrypt", "argon2", "not stated"]
    names = {"plaintext": "Plaintext", "md5": "MD5", "sha1": "SHA-1", "sha2": "SHA-2", "pbkdf2": "PBKDF2",
             "bcrypt": "bcrypt", "scrypt": "scrypt", "argon2": "Argon2", "not stated": "Not stated"}
    keys = [k for k in order if k in sc]
    fig, ax = plt.subplots(figsize=(8, 4))
    _hbar(ax, [names[k] for k in keys], [sc[k] for k in keys])
    fig.suptitle("How breached passwords were stored (weakest scheme named in HIBP description)", x=0.012, ha="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("Breaches that exposed passwords")
    figs["password_storage"] = _save(fig, "password_storage.png", figdir)

    lc = pa["length_counts_uniform_breached"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = sorted(int(k) for k in lc)
    tot = sum(lc.values())
    ax.bar(xs, [lc[k] / tot * 100 for k in xs], color=BLUE, width=0.8, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(xs, [str(x) if x < 20 else "20+" for x in xs])
    ax.set_title("Length of breached passwords (uniform wordlist sample)")
    ax.set_xlabel("Characters")
    ax.set_ylabel("% of passwords")
    ax.grid(axis="x", visible=False)
    figs["length_distribution"] = _save(fig, "length_distribution.png", figdir)

    pol = pa["policies"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _hbar(ax, [p["label"] for p in pol], [p["share_breached_uniform_pass_composition"] * 100 for p in pol], fmt="{:.1f}%")
    fig.suptitle("Breached passwords that pass each policy's length/composition rules", x=0.012, ha="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("% of breached passwords (uniform sample); blocklist check excluded")
    figs["policy_compliance"] = _save(fig, "policy_compliance.png", figdir)

    mods = metrics["models"]
    keys = list(MODEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    _hbar(axes[0], [MODEL_LABELS[k] for k in keys], [mods[k]["mae_log10"] for k in keys], fmt="{:.2f}")
    axes[0].set_title("Mean absolute error (orders of magnitude)")
    _hbar(axes[1], [MODEL_LABELS[k] for k in keys], [mods[k]["overestimate_10x_rate"] * 100 for k in keys], fmt="{:.0f}%", color=ORANGE)
    axes[1].set_title("Says >= 10x stronger than reality")
    figs["model_comparison"] = _save(fig, "model_comparison.png", figdir)

    processed = project_path(cfg["data"]["processed_dir"])
    test = pd.read_csv(processed / "test_predictions.csv", keep_default_na=False, dtype={"password": str})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True)
    lim = (0, max(18, float(test[["pred_charset_entropy_raw"]].max().iloc[0])))
    for ax, (col, title) in zip(axes, [("pred_charset_entropy_raw", "Charset entropy (raw)"),
                                       ("pred_pattern_estimator_raw", "Pattern estimator (raw)"),
                                       ("pred_gbm_full", "GBM: all features")]):
        ax.scatter(test["log10_rank"], test[col], s=9, alpha=0.35, color=BLUE, linewidths=0)
        ax.plot([0, 18], [0, 18], color=INK2, linewidth=1.2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Actual: log10 guess rank in HIBP")
        ax.set_xlim(0, 10.5)
        ax.set_ylim(*lim)
    axes[0].set_ylabel("Estimated log10 guesses")
    figs["estimate_vs_actual"] = _save(fig, "estimate_vs_actual.png", figdir)

    imp = metrics["permutation_importance_top15"][:12]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    _hbar(ax, [i["feature"] for i in imp], [i["mae_increase"] for i in imp], fmt="{:.3f}")
    fig.suptitle("Permutation importance (GBM, test split)", x=0.012, ha="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("Increase in MAE (log10) when shuffled")
    figs["feature_importance"] = _save(fig, "feature_importance.png", figdir)
    return figs


# --------------------------------------------------------------------------- markdown
def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def _big(n: float) -> str:
    for unit, size in (("billion", 1e9), ("million", 1e6)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:,.0f}"


def results_markdown(cfg: dict, br: dict, pa: dict, metrics: dict, meta: dict, figs: dict) -> str:
    m = metrics["models"]
    gf, cr, pe = m["gbm_full"], m["charset_entropy_raw"], m["pattern_estimator_raw"]
    L = []
    L.append("_Generated by `python -m pwanalysis.report`; every number comes from `outputs/summary.json` "
             "and `outputs/metrics.json`._\n")
    L.append("### Data collected\n")
    L.append(f"* **{br['n_breaches_total']:,} breaches** in HIBP's public breach list "
             f"({br['n_excluded_fabricated_or_spam']} fabricated or spam lists excluded from the stats below).")
    L.append(f"* **{meta['n_passwords']:,} wordlist passwords** looked up in Pwned Passwords via the k-anonymity API "
             f"({meta['n_log_uniform']:,} sampled log-uniformly by wordlist rank + {meta['n_random_bucket']:,} found in random buckets); "
             f"**{meta['n_breached']:,}** are in the corpus.")
    L.append(f"* **{meta['n_random_buckets']:,} random hash buckets** ({meta['n_corpus_sample_hashes']:,} hashes) sampled to estimate "
             f"the corpus size (**~{_big(metrics['corpus_size_estimate'])} distinct passwords**) and each password's guess rank.\n")

    L.append("### Breach landscape (HIBP metadata)\n")
    L.append(f"* {_pct(br['share_with_passwords'])} of real breaches ({br['n_with_passwords']} of {br['n_breaches']}) exposed passwords, "
             f"covering {_big(br['accounts_with_passwords'])} account records.")
    L.append(f"* Where HIBP's description says how passwords were stored ({br['n_stated_storage']} breaches), "
             f"**{_pct(br['share_stated_storage_plaintext_or_fast'])} were plaintext or a fast hash** (MD5 / SHA-1 / SHA-2): "
             f"{br['password_storage_counts'].get('plaintext', 0)} plaintext, {br['password_storage_counts'].get('md5', 0)} MD5, "
             f"{br['password_storage_counts'].get('sha1', 0)} SHA-1, against {br['password_storage_counts'].get('bcrypt', 0)} bcrypt.")
    L.append(f"* Median time from breach to appearing in HIBP: **{br['median_disclosure_lag_days']:.0f} days** "
             f"({br['median_disclosure_lag_days_with_passwords']:.0f} for breaches with passwords); "
             f"{_pct(br['share_disclosed_after_1_year'])} surfaced more than a year later.\n")
    L.append(f"![Breaches per year]({figs['breaches_per_year']})\n")
    L.append(f"![Password storage]({figs['password_storage']})\n")

    L.append("### How breached passwords are built\n")
    L.append(f"Uniform sample: {pa['uniform_sample_n']:,} wordlist passwords, of which "
             f"{_pct(pa['uniform_share_in_hibp'])} are in Pwned Passwords; median length of the breached ones "
             f"is {pa['median_length_uniform_breached']:.0f}.\n")
    L.append("| Pattern | Share of breached passwords |\n|---|---:|")
    for k, v in sorted(pa["pattern_prevalence_uniform_breached"].items(), key=lambda t: -t[1]):
        L.append(f"| {k} | {_pct(v)} |")
    L.append("\nTop hashcat-style masks (u=upper, l=lower, d=digit, s=symbol): " +
             ", ".join(f"`{x['mask']}` {_pct(x['share'])}" for x in pa["top_masks_uniform_breached"][:6]) + "\n")
    L.append(f"![Length distribution]({figs['length_distribution']})\n")

    L.append("### Password policy vs. reality\n")
    L.append("Pass rate from the uniform sample; median rank and crack share from all breached passwords that pass "
             "(log-uniform sample, larger n). With a breach blocklist, every one of these would be rejected.\n")
    L.append("| Policy | Breached passwords that pass its length/composition rules | Median guess rank of those | Cracked in < 1 day, offline bcrypt | Has breach blocklist |\n|---|---:|---:|---:|:---:|")
    for p in pa["policies"]:
        rank = f"{10 ** p['median_log10_rank_compliant']:,.0f}" if p["median_log10_rank_compliant"] is not None else "n/a"
        L.append(f"| {p['label']} | {_pct(p['share_breached_uniform_pass_composition'])} | {rank} | "
                 f"{_pct(p['share_compliant_cracked_1day_bcrypt'])} | {'yes' if p['blocklist'] else 'no'} |")
    leg = next(p for p in pa["policies"] if p["policy"] == "legacy_complexity")
    L.append(f"\nMost common breached passwords that satisfy legacy complexity rules: "
             + ", ".join(f"`{e}`" for e in leg["most_common_compliant_examples"]) + ".\n")
    L.append(f"![Policy compliance]({figs['policy_compliance']})\n")

    L.append("### Crack time for breached passwords\n")
    L.append("Uniform sample of breached passwords, using each one's empirical guess rank "
             "(an attacker replaying the corpus in frequency order):\n")
    L.append("| Scenario | Cracked within 1 second | Cracked within 1 day |\n|---|---:|---:|")
    for v in pa["crack_time_breached"].values():
        L.append(f"| {v['label']} | {_pct(v['share_within_1_second'])} | {_pct(v['share_within_1_day'])} |")
    if pa["famous_examples"]:
        L.append("\n| Password | Times in HIBP | Guess rank | Charset 'entropy' | Online throttled | Offline bcrypt |\n|---|---:|---:|---:|---|---|")
        sc = cfg["attack_scenarios"]
        for e in pa["famous_examples"]:
            lr = float(e["log10_rank"])
            L.append(f"| `{e['password']}` | {int(e['pwned_count']):,} | {10 ** lr:,.0f} | {float(e['charset_entropy_bits']):.0f} bits | "
                     f"{humanize(10 ** (lr - math.log10(sc['online_throttled']['rate'])))} | "
                     f"{humanize(10 ** (lr - math.log10(sc['offline_bcrypt_cost10']['rate'])))} |")
    L.append("")

    L.append("### Crack-time model\n")
    L.append(f"Target: log10 guess rank of {metrics['n_breached']:,} breached passwords "
             f"(train {metrics['n_train']:,} / test {metrics['n_test']:,}). Test-split results:\n")
    L.append("| Estimator | Spearman ρ | MAE (orders of magnitude) | Within 10x | Says >= 10x too strong |\n|---|---:|---:|---:|---:|")
    for k, lab in MODEL_LABELS.items():
        v = m[k]
        L.append(f"| {lab} | {v['spearman']:.3f} | {v['mae_log10']:.2f} | {_pct(v['within_1_order'])} | {_pct(v['overestimate_10x_rate'])} |")
    L.append(f"\n* Raw charset entropy overstates breached passwords by a median **10^{pa['entropy_overstatement_test_median_log10']:.1f}x** "
             f"(10^{pa['entropy_overstatement_test_legacy_compliant_median_log10']:.1f}x for ones that pass legacy complexity rules) "
             f"and calls {_pct(cr['overestimate_10x_rate'])} of them at least 10x stronger than they are.")
    L.append(f"* The pattern estimator alone cuts that to {_pct(pe['overestimate_10x_rate'])} (Spearman {pe['spearman']:.2f}); "
             f"the gradient-boosted model reaches Spearman **{gf['spearman']:.2f}**, MAE **{gf['mae_log10']:.2f}** orders of magnitude, "
             f"{_pct(gf['within_1_order'])} within 10x of the true rank.")
    imp = metrics["permutation_importance_top15"][:5]
    L.append("* Most important features: " + ", ".join(f"`{i['feature']}`" for i in imp) + ".")
    cc = metrics.get("censored_check")
    if cc:
        L.append(f"* Sanity check on {cc['n']} wordlist passwords that are *not* in HIBP: the model's median estimate is "
                 f"10^{cc['gbm_full_median_pred']:.1f} guesses vs 10^{cc['breached_test_median_true']:.1f} for breached test passwords "
                 f"(true value is above the corpus size, 10^{cc['corpus_log10']:.1f}; the model can only extrapolate so far — see Limitations).")
    L.append("")
    L.append(f"![Estimate vs actual]({figs['estimate_vs_actual']})\n")
    L.append(f"![Model comparison]({figs['model_comparison']})\n")
    L.append(f"![Feature importance]({figs['feature_importance']})\n")
    return "\n".join(L)


def inject_readme(block: str) -> None:
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
                 lambda _: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", text, flags=re.S)
    readme.write_text(new, encoding="utf-8")


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    processed = project_path(cfg["data"]["processed_dir"])
    outputs = project_path(cfg["artifacts"]["outputs_dir"])
    metrics = json.loads((outputs / "metrics.json").read_text())
    meta = json.loads((processed / "dataset_meta.json").read_text())
    br = analyse_breaches(json.loads((processed / "breaches.json").read_text()))
    pa = analyse_passwords(cfg, metrics)
    figs = make_figures(cfg, br, pa, metrics)
    summary = {"dataset": meta, "breaches": br, "passwords": pa}
    (outputs / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    block = results_markdown(cfg, br, pa, metrics, meta, figs)
    (outputs / "report.md").write_text("# Password Breach & Strength Analysis: results\n\n" +
                                       block.replace("(outputs/figures/", "(figures/"), encoding="utf-8")
    inject_readme(block)
    log.info("wrote outputs/summary.json, outputs/report.md, %d figures", len(figs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
