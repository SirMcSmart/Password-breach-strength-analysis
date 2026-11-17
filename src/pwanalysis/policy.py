"""Password-policy compliance checks (policies are defined in config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def char_classes(password: str) -> dict[str, bool]:
    return {
        "lower": any(c.islower() for c in password),
        "upper": any(c.isupper() for c in password),
        "digit": any(c.isdigit() for c in password),
        "symbol": any(not c.isalnum() for c in password),
    }


@dataclass
class PolicyResult:
    policy: str
    label: str
    compliant: bool
    failures: list[str]


def check_policy(password: str, name: str, rules: dict[str, Any], pwned_count: int | None = None) -> PolicyResult:
    """Check one policy. ``pwned_count=None`` means the blocklist check was not performed (skipped)."""
    failures: list[str] = []
    classes = char_classes(password)
    n_classes = sum(classes.values())
    if rules.get("min_length") and len(password) < rules["min_length"]:
        failures.append(f"shorter than {rules['min_length']} characters")
    if rules.get("min_classes") and n_classes < rules["min_classes"]:
        failures.append(f"uses {n_classes} of 4 character classes, needs {rules['min_classes']}")
    for req in rules.get("required_classes") or []:
        ok = (classes["lower"] or classes["upper"]) if req == "alpha" else classes[req]
        if not ok:
            failures.append(f"missing required class: {req}")
    if rules.get("blocklist") and pwned_count is not None and pwned_count > 0:
        failures.append(f"appears in breach corpus ({pwned_count:,} times)")
    return PolicyResult(name, rules.get("label", name), not failures, failures)


def check_all(password: str, policies: dict[str, dict], pwned_count: int | None = None) -> list[PolicyResult]:
    return [check_policy(password, n, r, pwned_count) for n, r in policies.items()]


def composition_compliant(password: str, rules: dict[str, Any]) -> bool:
    """Compliance with the length/composition rules only (blocklist ignored)."""
    return check_policy(password, "", {**rules, "blocklist": False}).compliant
