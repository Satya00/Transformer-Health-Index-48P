"""Trend-based hard-capping rules for the dynamic HI model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd


TREND_CAPPING_SHEET = "Trend_Based_Capping_48"


def _number(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(numeric) else float(numeric)


def load_trend_capping_rules(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        df = pd.read_excel(path, sheet_name=TREND_CAPPING_SHEET)
    except ValueError:
        return []
    rules = []
    for row in df.to_dict("records"):
        status = str(row.get("Rule_Status", "")).lower()
        if "active" not in status:
            continue
        rules.append(
            {
                "parameter": str(row["Parameter"]).strip(),
                "bad_direction": str(row.get("Bad_Direction", "")).lower(),
                "one_year_pct": _number(row.get("1-Year Worsening % for HI=0")),
                "two_year_pct": _number(row.get("2-Year Cumulative Worsening % for HI=0")),
                "denominator_floor": _number(row.get("Denominator Floor"), 1.0),
                "minimum_absolute_change": _number(row.get("Minimum Absolute Change")),
                "activation_gate": str(row.get("Activation Gate", "")).strip(),
                "reason": str(row.get("Technical Reason", "")).strip(),
            }
        )
    return rules


def _gate_passes(rule: Mapping[str, object], current: float, previous: float) -> bool:
    text = str(rule.get("activation_gate", ""))
    if "not applicable" in text.lower():
        return False
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return True
    threshold = numbers[0]
    lowered = text.lower()
    if "≤" in text or "<=" in text:
        current_ok = current <= threshold
    elif "≥" in text or ">=" in text:
        current_ok = current >= threshold
    else:
        current_ok = True
    if "previous year" in lowered and len(numbers) > 1:
        previous_ok = previous >= numbers[1]
    else:
        previous_ok = True
    return current_ok and previous_ok


def _risk_value(value: float, bad_direction: str) -> float:
    if "absolute" in bad_direction:
        return abs(value)
    return value


def _worsening(previous: float, current: float, floor: float, bad_direction: str) -> tuple[float, float]:
    previous_risk = _risk_value(previous, bad_direction)
    current_risk = _risk_value(current, bad_direction)
    if "lower" in bad_direction and "absolute" not in bad_direction:
        delta = previous - current
    else:
        delta = current_risk - previous_risk
    denominator = max(abs(previous_risk), floor, 1e-9)
    return (delta / denominator) * 100.0, abs(delta)


def evaluate_trend_capping(values: Mapping[str, float], rules: list[dict]) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for rule in rules:
        parameter = str(rule["parameter"])
        old_key = f"{parameter}_Tminus2"
        previous_key = f"{parameter}_Tminus1"
        current_key = f"{parameter}_T0_Current"
        if old_key not in values or previous_key not in values or current_key not in values:
            continue
        old = float(values[old_key])
        previous = float(values[previous_key])
        current = float(values[current_key])
        if not _gate_passes(rule, current, previous):
            continue

        bad_direction = str(rule["bad_direction"])
        latest_pct, latest_abs = _worsening(previous, current, float(rule["denominator_floor"]), bad_direction)
        previous_pct, _ = _worsening(old, previous, float(rule["denominator_floor"]), bad_direction)
        total_pct, total_abs = _worsening(old, current, float(rule["denominator_floor"]), bad_direction)
        min_abs = float(rule["minimum_absolute_change"])
        absolute_change_ok = max(latest_abs, total_abs) >= min_abs
        latest_worse = latest_pct > 0.0
        previous_worse = previous_pct > 0.0
        one_year = float(rule["one_year_pct"])
        two_year = float(rule["two_year_pct"])
        triggered = absolute_change_ok and (
            (latest_pct >= one_year and previous_worse)
            or (total_pct >= two_year and latest_worse)
            or (latest_pct >= 1.5 * one_year)
        )
        if triggered:
            violations.append(
                {
                    "parameter": parameter,
                    "current": current,
                    "previous": previous,
                    "old": old,
                    "latest_pct": latest_pct,
                    "total_pct": total_pct,
                    "reason": str(rule.get("reason", "")),
                }
            )
    return violations
