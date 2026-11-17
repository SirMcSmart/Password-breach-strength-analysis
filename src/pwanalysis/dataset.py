"""Build the analysis dataset.

1. Download the plaintext candidate wordlist and the ranked English word list (text files only;
   they are read as data, never executed).
2. Download HIBP breach metadata.
3. Sample candidate passwords log-uniformly by wordlist rank and pick random k-anonymity buckets.
4. Query every needed bucket once (cached in ``data/cache/ranges.jsonl``, so re-runs are resumable)
   and resolve *every* wordlist password whose hash falls in a queried bucket. Passwords found
   through the random buckets are a uniform sample of the wordlist and are tagged ``random_bucket``.
5. Write ``data/processed/passwords.csv``, ``corpus_sample_counts.npy`` and ``breaches.json``.

Usage: ``python -m pwanalysis.dataset``
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import load_config, project_path
from .hibp import HIBPClient, sha1_hex
from .rank import N_BUCKETS

log = logging.getLogger(__name__)


def download(url: str, dest: Path, user_agent: str) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    r = requests.get(url, headers={"User-Agent": user_agent}, timeout=120)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest


def read_wordlist(path: Path) -> pd.DataFrame:
    """Frequency-ordered wordlist -> DataFrame(password, wl_rank). First occurrence wins."""
    rows, seen = [], set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            pw = line.rstrip("\r\n")
            if not pw or pw in seen or "�" in pw or not pw.isprintable():
                continue
            seen.add(pw)
            rows.append(pw)
    return pd.DataFrame({"password": rows, "wl_rank": np.arange(1, len(rows) + 1)})


def log_uniform_ranks(n_total: int, n: int, rng: np.random.Generator, head: int = 100) -> np.ndarray:
    """Unique 1-based ranks: all of the top ``head`` plus a log-uniform draw over the rest."""
    chosen = set(range(1, min(head, n_total) + 1))
    target = min(n, n_total)
    while len(chosen) < target:
        u = rng.uniform(np.log(head + 1), np.log(n_total + 1), size=2 * (target - len(chosen)))
        chosen.update(np.floor(np.exp(u)).astype(int).clip(1, n_total).tolist())
    ranks = np.array(sorted(chosen))
    if len(ranks) > target:
        keep_head = ranks[ranks <= head]
        tail = rng.choice(ranks[ranks > head], size=target - len(keep_head), replace=False)
        ranks = np.sort(np.concatenate([keep_head, tail]))
    return ranks


def load_range_cache(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["prefix"]] = rec
    return done


def build(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    dcfg, scfg = cfg["data"], cfg["sample"]
    raw, cache, processed = (project_path(dcfg[k]) for k in ("raw_dir", "cache_dir", "processed_dir"))
    for d in (raw, cache, processed):
        d.mkdir(parents=True, exist_ok=True)
    ua = cfg["hibp"]["user_agent"]
    client = HIBPClient(cfg)
    rng = np.random.default_rng(cfg["seed"])

    wl_path = download(dcfg["wordlist_url"], raw / dcfg["wordlist_name"], ua)
    download(dcfg["english_url"], raw / dcfg["english_name"], ua)

    breaches_path = processed / "breaches.json"
    if not breaches_path.exists():
        breaches_path.write_text(json.dumps(client.fetch_breaches(), indent=1), encoding="utf-8")
    log.info("breach metadata: %d breaches", len(json.loads(breaches_path.read_text())))

    wl = read_wordlist(wl_path)
    log.info("wordlist: %d unique passwords", len(wl))
    hashes = wl["password"].map(sha1_hex)
    wl["prefix"], wl["suffix"] = hashes.str[:5], hashes.str[5:]

    sampled_ranks = log_uniform_ranks(len(wl), scfg["n_passwords"], rng)
    sampled_prefixes = set(wl.loc[wl["wl_rank"].isin(sampled_ranks), "prefix"])
    random_prefixes = {f"{i:05X}" for i in rng.choice(N_BUCKETS, size=scfg["n_random_buckets"], replace=False)}
    by_prefix = wl.groupby("prefix")["suffix"].apply(set).to_dict()

    cache_path = cache / "ranges.jsonl"
    done = load_range_cache(cache_path)
    needed = sampled_prefixes | random_prefixes
    todo = sorted(p for p in needed if p not in done or (p in random_prefixes and "counts" not in done[p]))
    log.info("buckets: %d needed, %d cached, %d to fetch", len(needed), len(done), len(todo))

    def work(prefix: str) -> dict:
        rng_bucket = client.fetch_range(prefix)
        rec = {"prefix": prefix, "hits": {s: rng_bucket[s] for s in by_prefix.get(prefix, ()) if s in rng_bucket}}
        if prefix in random_prefixes:
            rec["counts"] = sorted(rng_bucket.values())
        return rec

    with open(cache_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(cfg["hibp"]["workers"]) as pool:
        futures = [pool.submit(work, p) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            done[rec["prefix"]] = rec
            out.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                out.flush()
                log.info("  fetched %d/%d", i, len(todo))

    queried = set(done)
    hits = {(p, s): n for p, rec in done.items() for s, n in rec["hits"].items()}
    resolved = wl[wl["prefix"].isin(queried)].copy()
    resolved["pwned_count"] = [hits.get((p, s), 0) for p, s in zip(resolved["prefix"], resolved["suffix"])]
    in_sample = resolved["wl_rank"].isin(sampled_ranks)
    in_random = resolved["prefix"].isin(random_prefixes)
    resolved = resolved[in_sample | in_random].copy()
    resolved["source"] = np.where(resolved["wl_rank"].isin(sampled_ranks), "log_uniform", "random_bucket")
    resolved["in_random_bucket"] = resolved["prefix"].isin(random_prefixes)
    resolved = resolved.drop(columns=["suffix"]).sort_values("wl_rank")
    resolved.to_csv(processed / "passwords.csv", index=False)

    counts = np.concatenate([np.asarray(done[p]["counts"], dtype=np.int64) for p in sorted(random_prefixes)])
    np.save(processed / "corpus_sample_counts.npy", counts)
    (processed / "dataset_meta.json").write_text(json.dumps({
        "wordlist_unique": int(len(wl)),
        "n_passwords": int(len(resolved)),
        "n_log_uniform": int((resolved["source"] == "log_uniform").sum()),
        "n_random_bucket": int((resolved["source"] == "random_bucket").sum()),
        "n_breached": int((resolved["pwned_count"] > 0).sum()),
        "n_random_buckets": len(random_prefixes),
        "n_corpus_sample_hashes": int(len(counts)),
        "n_buckets_queried": len(queried),
    }, indent=1), encoding="utf-8")
    log.info("wrote %d passwords (%d breached), %d corpus sample hashes",
             len(resolved), int((resolved["pwned_count"] > 0).sum()), len(counts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
