"""Analysis of HIBP breach metadata (``/api/v3/breaches``).

Password storage is inferred from each breach's free-text description (e.g. "salted MD5 hashes",
"bcrypt"). That is a heuristic: descriptions are written by HIBP, not every one says how passwords
were stored, and a breach that mentions several schemes is classed by the weakest one named.
"""

from __future__ import annotations

import html
import re
from collections import Counter

import pandas as pd

# ordered weakest -> strongest; the weakest scheme mentioned wins
STORAGE_PATTERNS = [
    ("plaintext", r"plain[\s-]?text|plaintext|clear[\s-]?text|unhashed|unencrypted password"),
    ("md5", r"\bmd5\b"),
    ("sha1", r"\bsha[\s-]?1\b"),
    ("sha2", r"\bsha[\s-]?(?:2|256|384|512)\b"),
    ("pbkdf2", r"\bpbkdf2\b"),
    ("scrypt", r"\bscrypt\b"),
    ("bcrypt", r"\bbcrypt\b"),
    ("argon2", r"\bargon2"),
]
STORAGE_GROUP = {"plaintext": "Plaintext", "md5": "Fast hash", "sha1": "Fast hash", "sha2": "Fast hash",
                 "pbkdf2": "Slow hash", "scrypt": "Slow hash", "bcrypt": "Slow hash", "argon2": "Slow hash",
                 None: "Not stated"}


def strip_html(s: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or ""))


def password_storage(description: str | None) -> str | None:
    text = strip_html(description).lower()
    for name, pat in STORAGE_PATTERNS:
        if re.search(pat, text):
            return name
    return None


def to_frame(breaches: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(breaches)
    df["breach_date"] = pd.to_datetime(df["BreachDate"], errors="coerce")
    df["added_date"] = pd.to_datetime(df["AddedDate"], errors="coerce", utc=True).dt.tz_localize(None)
    df["year"] = df["breach_date"].dt.year
    df["has_passwords"] = df["DataClasses"].apply(lambda c: "Passwords" in (c or []))
    df["disclosure_lag_days"] = (df["added_date"] - df["breach_date"]).dt.days
    storage = [password_storage(d) if hp else None for d, hp in zip(df["Description"], df["has_passwords"])]
    df["storage"] = storage
    df["storage_group"] = [STORAGE_GROUP[s] if hp else None for s, hp in zip(storage, df["has_passwords"])]
    for col in ("IsFabricated", "IsSpamList", "IsVerified", "IsSensitive", "IsMalware", "IsStealerLog"):
        if col not in df:
            df[col] = False
    return df


def analyse(breaches: list[dict]) -> dict:
    df_all = to_frame(breaches)
    # Fabricated breaches and spam lists are not real credential compromises.
    df = df_all[~df_all["IsFabricated"].astype(bool) & ~df_all["IsSpamList"].astype(bool)].copy()
    pw = df[df["has_passwords"]]

    per_year = (df.groupby("year")
                  .agg(breaches=("Name", "size"), with_passwords=("has_passwords", "sum"),
                       accounts=("PwnCount", "sum"),
                       median_lag_days=("disclosure_lag_days", "median"))
                  .reset_index())
    per_year["year"] = per_year["year"].astype(int)
    pw_accounts = pw.groupby("year")["PwnCount"].sum()
    per_year["password_accounts"] = per_year["year"].map(pw_accounts).fillna(0).astype(int)

    classes = Counter(c for cl in df["DataClasses"] for c in (cl or []))
    storage_counts = pw["storage"].fillna("not stated").value_counts().to_dict()
    storage_groups = pw["storage_group"].value_counts().to_dict()
    storage_group_accounts = pw.groupby("storage_group")["PwnCount"].sum().to_dict()
    stated = pw[pw["storage"].notna()]

    largest = (pw.sort_values("PwnCount", ascending=False).head(10)
                 [["Title", "BreachDate", "PwnCount", "storage"]]
                 .rename(columns={"Title": "title", "BreachDate": "breach_date", "PwnCount": "accounts"}))
    largest["storage"] = largest["storage"].fillna("not stated")

    return {
        "n_breaches_total": int(len(df_all)),
        "n_excluded_fabricated_or_spam": int(len(df_all) - len(df)),
        "n_breaches": int(len(df)),
        "n_with_passwords": int(len(pw)),
        "share_with_passwords": float(len(pw) / len(df)),
        "accounts_total": int(df["PwnCount"].sum()),
        "accounts_with_passwords": int(pw["PwnCount"].sum()),
        "n_verified": int(df["IsVerified"].astype(bool).sum()),
        "median_disclosure_lag_days": float(df["disclosure_lag_days"].median()),
        "median_disclosure_lag_days_with_passwords": float(pw["disclosure_lag_days"].median()),
        "share_disclosed_after_1_year": float((df["disclosure_lag_days"] > 365).mean()),
        "per_year": per_year.to_dict(orient="records"),
        "top_data_classes": [{"data_class": k, "breaches": v} for k, v in classes.most_common(15)],
        "password_storage_counts": {str(k): int(v) for k, v in storage_counts.items()},
        "password_storage_groups": {str(k): int(v) for k, v in storage_groups.items()},
        "password_storage_group_accounts": {str(k): int(v) for k, v in storage_group_accounts.items()},
        "share_stated_storage_plaintext_or_fast": float(stated["storage_group"].isin(["Plaintext", "Fast hash"]).mean()) if len(stated) else None,
        "n_stated_storage": int(len(stated)),
        "largest_password_breaches": largest.to_dict(orient="records"),
    }
