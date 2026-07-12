"""Dynamic Transformer Health Monitoring System web app."""

from __future__ import annotations

import json
import html
import io
import os
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
import mimetypes

import numpy as np
import pandas as pd

import health_rules
from health_rules import RAW_FEATURES, build_feature_row, gas_conditions, gas_scores, label_rule_features, label_rule_name, operating_scores, rule_summary
from train_model import DEFAULT_DATASET, INPUT_FEATURES, MODEL_DIR, OUTPUT_DIR, configure_rules_from_workbook, predict
from trend_capping import evaluate_trend_capping, load_trend_capping_rules


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
LOGO_PATH = STATIC_DIR / "power_asset_logo.png"
LOGO_URL = "/static/power_asset_logo.png"
COVER_TEMPLATE_PATH = STATIC_DIR / "report_cover_template.pdf"
COVER_ICON_TRANSFORMER_PATH = STATIC_DIR / "cover_icon_transformer.png"
COVER_ICON_DATE_PATH = STATIC_DIR / "cover_icon_date.png"
COVER_ICON_REPORT_ID_PATH = STATIC_DIR / "cover_icon_report_id.png"
MODEL_PATH = MODEL_DIR / "health_index_model.npz"
DYNAMIC_MODEL_PATH = MODEL_DIR / "dynamic_health_index_model.npz"
METADATA_PATH = MODEL_DIR / "metadata.json"
DYNAMIC_FEATURES_PATH = MODEL_DIR / "dynamic_features.json"
CAPPING_REASONS_PATH = APP_DIR / "data" / "capping_with_reasons.xlsx"
AGE_HARD_CAPPING_PATH = APP_DIR / "data" / "age_wise_hard_capping_with_reasons.xlsx"
DYNAMIC_DATASET_PATH = APP_DIR / "data" / "Transformer_Dynamic_HI_3Year_48_Parameter_Dataset.xlsx"
TREND_CAPPING_PATH = APP_DIR / "data" / "Transformer_HI_48_Trend_Based_Capping.xlsx"
TREND_REASONS_PATH = APP_DIR / "data" / "Transformer_Dynamic_HI_Trend_Reasons_48_Parameters.xlsx"
configure_rules_from_workbook(DEFAULT_DATASET)


DEFAULT_INPUT = {
    "H2": 265.8,
    "Methane": 148.1,
    "Ethane": 75.1,
    "Ethylene": 76.6,
    "Acetylene": 1.6,
    "CO": 382.9,
    "CO2": 2967.7,
    "BDV": 58.8,
    "Moisture_PPM": 20.8,
    "WTI_HV": 79.9,
    "WTI_LV": 85.9,
    "OTI": 75.7,
    "Loading_pct": 89.3,
    "Ambient_Temp": 37.6,
    "Age_Years": 10.8,
    "IR_HV_E_1min": 10000.0,
    "IR_LV_E_1min": 10000.0,
    "IR_TV_E_1min": 9874.9,
    "IR_HV_E_10min": 18982.5,
    "IR_LV_E_10min": 20000.0,
    "IR_TV_E_10min": 19749.8,
    "PI_HV": 1.898,
    "PI_LV": 2.0,
    "PI_TV": 2.0,
    "Winding_HV_TanDelta_pct": 0.5,
    "Winding_LV_TanDelta_pct": 0.509,
    "Winding_TV_TanDelta_pct": 0.511,
    "Winding_HV_CapChange_pct": 2.25,
    "Winding_LV_CapChange_pct": 2.0,
    "Winding_TV_CapChange_pct": 2.19,
    "Bushing_HV_R_TanDelta_pct": 0.54,
    "Bushing_HV_Y_TanDelta_pct": 0.5,
    "Bushing_HV_B_TanDelta_pct": 0.571,
    "Bushing_HV_R_CapChange_pct": 2.11,
    "Bushing_HV_Y_CapChange_pct": 2.0,
    "Bushing_HV_B_CapChange_pct": -2.53,
    "Bushing_LV_R_TanDelta_pct": 0.541,
    "Bushing_LV_Y_TanDelta_pct": 0.577,
    "Bushing_LV_B_TanDelta_pct": 0.552,
    "Bushing_LV_R_CapChange_pct": 2.0,
    "Bushing_LV_Y_CapChange_pct": 2.72,
    "Bushing_LV_B_CapChange_pct": 3.02,
    "Bushing_TV_R_TanDelta_pct": 0.546,
    "Bushing_TV_Y_TanDelta_pct": 0.5,
    "Bushing_TV_B_TanDelta_pct": 0.523,
    "Bushing_TV_R_CapChange_pct": 2.69,
    "Bushing_TV_Y_CapChange_pct": 2.55,
    "Bushing_TV_B_CapChange_pct": 2.13,
}

TEXT_INPUT_DEFAULTS = {
    "SubstationName": "Substation-1",
    "VoltageRatio": "220/132 kV",
}

INPUT_LABELS = {
    "H2": ("H2 Hydrogen", "ppm"),
    "Methane": ("CH4 Methane", "ppm"),
    "Ethane": ("C2H6 Ethane", "ppm"),
    "Ethylene": ("C2H4 Ethylene", "ppm"),
    "Acetylene": ("C2H2 Acetylene", "ppm"),
    "CO": ("CO Carbon Monoxide", "ppm"),
    "CO2": ("CO2 Carbon Dioxide", "ppm"),
    "BDV": ("BDV", "kV"),
    "WTI_HV": ("WTI HV", "deg C"),
    "WTI_LV": ("WTI LV", "deg C"),
    "OTI": ("Maximum OTI", "deg C"),
    "Loading_pct": ("Loading", "%"),
    "Ambient_Temp": ("Ambient Temperature", "deg C"),
    "Age_Years": ("Age", "years"),
    "Moisture_PPM": ("Moisture", "ppm"),
    "IR_HV_E_1min": ("IR HV-E 1 min", "Mohm"),
    "IR_LV_E_1min": ("IR LV-E 1 min", "Mohm"),
    "IR_TV_E_1min": ("IR TV-E 1 min", "Mohm"),
    "IR_HV_E_10min": ("IR HV-E 10 min", "Mohm"),
    "IR_LV_E_10min": ("IR LV-E 10 min", "Mohm"),
    "IR_TV_E_10min": ("IR TV-E 10 min", "Mohm"),
    "PI_HV": ("PI HV", ""),
    "PI_LV": ("PI LV", ""),
    "PI_TV": ("PI TV", ""),
    "Winding_HV_TanDelta_pct": ("Winding HV Tan Delta", "%"),
    "Winding_LV_TanDelta_pct": ("Winding LV Tan Delta", "%"),
    "Winding_TV_TanDelta_pct": ("Winding TV Tan Delta", "%"),
    "Winding_HV_CapChange_pct": ("Winding HV Cap Change", "%"),
    "Winding_LV_CapChange_pct": ("Winding LV Cap Change", "%"),
    "Winding_TV_CapChange_pct": ("Winding TV Cap Change", "%"),
    "Bushing_HV_R_TanDelta_pct": ("Bushing HV-R Tan Delta", "%"),
    "Bushing_HV_Y_TanDelta_pct": ("Bushing HV-Y Tan Delta", "%"),
    "Bushing_HV_B_TanDelta_pct": ("Bushing HV-B Tan Delta", "%"),
    "Bushing_HV_R_CapChange_pct": ("Bushing HV-R Cap Change", "%"),
    "Bushing_HV_Y_CapChange_pct": ("Bushing HV-Y Cap Change", "%"),
    "Bushing_HV_B_CapChange_pct": ("Bushing HV-B Cap Change", "%"),
    "Bushing_LV_R_TanDelta_pct": ("Bushing LV-R Tan Delta", "%"),
    "Bushing_LV_Y_TanDelta_pct": ("Bushing LV-Y Tan Delta", "%"),
    "Bushing_LV_B_TanDelta_pct": ("Bushing LV-B Tan Delta", "%"),
    "Bushing_LV_R_CapChange_pct": ("Bushing LV-R Cap Change", "%"),
    "Bushing_LV_Y_CapChange_pct": ("Bushing LV-Y Cap Change", "%"),
    "Bushing_LV_B_CapChange_pct": ("Bushing LV-B Cap Change", "%"),
    "Bushing_TV_R_TanDelta_pct": ("Bushing TV-R Tan Delta", "%"),
    "Bushing_TV_Y_TanDelta_pct": ("Bushing TV-Y Tan Delta", "%"),
    "Bushing_TV_B_TanDelta_pct": ("Bushing TV-B Tan Delta", "%"),
    "Bushing_TV_R_CapChange_pct": ("Bushing TV-R Cap Change", "%"),
    "Bushing_TV_Y_CapChange_pct": ("Bushing TV-Y Cap Change", "%"),
    "Bushing_TV_B_CapChange_pct": ("Bushing TV-B Cap Change", "%"),
}

INPUT_SECTIONS = (
    ("DGA Inputs", ("H2", "Methane", "Ethane", "Ethylene", "Acetylene", "CO", "CO2")),
    ("Oil, Temperature & Loading", ("BDV", "Moisture_PPM", "WTI_HV", "WTI_LV", "OTI", "Loading_pct", "Ambient_Temp")),
    ("IR / PI Inputs", ("IR_HV_E_1min", "IR_LV_E_1min", "IR_TV_E_1min", "IR_HV_E_10min", "IR_LV_E_10min", "IR_TV_E_10min", "PI_HV", "PI_LV", "PI_TV")),
    ("Winding Tan Delta / Capacitance", ("Winding_HV_TanDelta_pct", "Winding_LV_TanDelta_pct", "Winding_TV_TanDelta_pct", "Winding_HV_CapChange_pct", "Winding_LV_CapChange_pct", "Winding_TV_CapChange_pct")),
    ("Bushing Tan Delta / Capacitance", ("Bushing_HV_R_TanDelta_pct", "Bushing_HV_Y_TanDelta_pct", "Bushing_HV_B_TanDelta_pct", "Bushing_HV_R_CapChange_pct", "Bushing_HV_Y_CapChange_pct", "Bushing_HV_B_CapChange_pct", "Bushing_LV_R_TanDelta_pct", "Bushing_LV_Y_TanDelta_pct", "Bushing_LV_B_TanDelta_pct", "Bushing_LV_R_CapChange_pct", "Bushing_LV_Y_CapChange_pct", "Bushing_LV_B_CapChange_pct", "Bushing_TV_R_TanDelta_pct", "Bushing_TV_Y_TanDelta_pct", "Bushing_TV_B_TanDelta_pct", "Bushing_TV_R_CapChange_pct", "Bushing_TV_Y_CapChange_pct", "Bushing_TV_B_CapChange_pct")),
)

TIME_POINT_LABELS = {
    "Tminus2": "Year 1 (Oldest)",
    "Tminus1": "Year 2",
    "T0_Current": "Year 3 (Current)",
}

TIME_POINT_YEAR_FIELD = {
    "Tminus2": "TrendYear_Tminus2",
    "Tminus1": "TrendYear_Tminus1",
    "T0_Current": "TrendYear_T0_Current",
}

TIME_POINT_DEFAULT_YEAR = {
    "Tminus2": "2024",
    "Tminus1": "2025",
    "T0_Current": "2026",
}

DYNAMIC_COLUMN_TO_CURRENT = {
    "CH4": "Methane",
    "C2H6": "Ethane",
    "C2H4": "Ethylene",
    "C2H2": "Acetylene",
}


GAS_REASON_LABELS = {
    "H₂": "H2",
    "CH₄": "Methane",
    "C₂H₆": "Ethane",
    "C₂H₄": "Ethylene",
    "C₂H₂": "Acetylene",
    "CO": "CO",
    "CO₂": "CO2",
}


def parse_range(text: object) -> tuple[float | None, float | None]:
    value = str(text).strip().replace(",", "")
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return None, None
    if value.startswith(">"):
        return numbers[0], None
    if value.startswith("<"):
        return None, numbers[0]
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers), max(numbers)


def value_in_range(value: float, range_text: object) -> bool:
    text = str(range_text).strip()
    low, high = parse_range(text)
    if low is not None and high is None:
        return value > low
    if low is None and high is not None:
        return value < high
    if low is not None and high is not None:
        return low <= value <= high
    return False


def load_capping_reason_rules() -> dict[str, dict]:
    gas_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=0).dropna(subset=["Gas (ppm)"])
    temp_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=1)
    bdv_df = pd.read_excel(CAPPING_REASONS_PATH, sheet_name=2)

    gas_rules = {}
    for record in gas_df.to_dict("records"):
        gas_label = str(record["Gas (ppm)"]).strip()
        column = GAS_REASON_LABELS.get(gas_label)
        if not column:
            continue
        gas_rules[column] = {
            "poor_range": record["value1 "],
            "poor_reason": str(record[" Reason"]).strip(),
            "critical_range": record["value 2"],
            "critical_reason": str(record["Reason"]).strip(),
        }

    temp_rules = {str(row["Parameter"]).split()[0]: row for row in temp_df.to_dict("records")}
    bdv_rules = bdv_df.to_dict("records")
    return {"gas": gas_rules, "temperature": temp_rules, "bdv": bdv_rules}


CAPPING_REASON_RULES = load_capping_reason_rules()


AGE_BANDS = (
    ("0–10 yr", 0.0, 10.0),
    ("11–20 yr", 10.0, 20.0),
    ("21–30 yr", 20.0, 30.0),
    ("31–40 yr", 30.0, 40.0),
    (">40 yr", 40.0, float("inf")),
)


def normalize_rule_parameter(value: object) -> str:
    return (
        str(value)
        .strip()
        .replace("₂", "2")
        .replace("₄", "4")
        .replace("₆", "6")
        .replace("δ", "delta")
        .lower()
    )


def load_age_hard_caps() -> dict[str, dict]:
    if not AGE_HARD_CAPPING_PATH.exists():
        return {}
    df = pd.read_excel(AGE_HARD_CAPPING_PATH)
    rules = {}
    for record in df.to_dict("records"):
        key = normalize_rule_parameter(record.get("Parameter", ""))
        if not key:
            continue
        rules[key] = {
            "label": str(record["Parameter"]).strip(),
            "reason": str(record.get("Reason", "")).strip(),
            "limits": {
                band: float(record[band])
                for band, _, _ in AGE_BANDS
                if band in record and pd.notna(record[band])
            },
        }
    return rules


AGE_HARD_CAP_RULES = load_age_hard_caps()


def parse_prediction_values(query: dict[str, list[str]]) -> dict[str, float]:
    return {key: float(query.get(key, [DEFAULT_INPUT[key]])[0]) for key in DEFAULT_INPUT}


def safe_pi(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return numerator / denominator


def add_calculated_inputs(values: dict[str, float]) -> dict[str, float]:
    enriched = dict(values)
    enriched["PI_HV"] = safe_pi(enriched["IR_HV_E_10min"], enriched["IR_HV_E_1min"])
    enriched["PI_LV"] = safe_pi(enriched["IR_LV_E_10min"], enriched["IR_LV_E_1min"])
    enriched["PI_TV"] = safe_pi(enriched["IR_TV_E_10min"], enriched["IR_TV_E_1min"])
    enriched["WTI"] = max(float(enriched.get("WTI_HV", 0.0)), float(enriched.get("WTI_LV", 0.0)))
    return enriched


def pi_condition(value: float) -> str:
    if value < 1.0:
        return "Hard Cap"
    if value < 1.25:
        return "Very Poor"
    if value < 1.5:
        return "Poor"
    if value < 2.0:
        return "Fair"
    return "Good"


def query_text(query: dict[str, list[str]], key: str, default: str = "") -> str:
    return str(query.get(key, [default])[0])


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "substation"


def public_base_url() -> str:
    return (
        os.environ.get("PUBLIC_BASE_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def hidden_form_fields(query: dict[str, list[str]]) -> str:
    fields = []
    for key, values in query.items():
        value = values[0] if values else ""
        fields.append(f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(str(value))}">')
    return "\n".join(fields)


def parse_form_body(handler: BaseHTTPRequestHandler) -> dict[str, list[str]]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(length).decode("utf-8") if length else ""
    return parse_qs(body)


def load_model() -> dict[str, np.ndarray]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train_model.py")
    loaded = np.load(MODEL_PATH)
    return {name: loaded[name] for name in loaded.files}


def load_dynamic_model() -> dict[str, np.ndarray]:
    if not DYNAMIC_MODEL_PATH.exists():
        raise FileNotFoundError("Dynamic model not found. Run: python train_dynamic_model.py")
    loaded = np.load(DYNAMIC_MODEL_PATH)
    return {name: loaded[name] for name in loaded.files}


def dynamic_features() -> list[str]:
    if DYNAMIC_FEATURES_PATH.exists():
        return json.loads(DYNAMIC_FEATURES_PATH.read_text(encoding="utf-8"))
    feature_map = pd.read_excel(DYNAMIC_DATASET_PATH, sheet_name="Dynamic_Feature_Map")
    feature_map = feature_map[feature_map["Use_for_Training"].astype(str).str.lower().eq("yes")]
    return [str(value) for value in feature_map["Model_Column"].tolist()]


DYNAMIC_INPUT_FEATURES = dynamic_features()


def dynamic_default_input() -> dict[str, float]:
    source_path = OUTPUT_DIR / "Dynamic_HI_Training_Set.csv"
    if source_path.exists():
        df = pd.read_csv(source_path)
    else:
        df = pd.read_excel(DYNAMIC_DATASET_PATH, sheet_name="Dynamic_Training_Data")
    if "Record_ID" in df.columns and (pd.to_numeric(df["Record_ID"], errors="coerce") == 1918).any():
        row = df.loc[pd.to_numeric(df["Record_ID"], errors="coerce") == 1918].iloc[0]
    else:
        target = pd.to_numeric(df.get("DHI_training_target", df.get("Dynamic_HI_Final")), errors="coerce")
        if float(target.max()) <= 1.5:
            target = target * 100.0
        candidates = df[(target >= 68.0) & (target <= 72.0)].copy()
        if not candidates.empty:
            row = candidates.iloc[(target.loc[candidates.index] - 70.0).abs().argsort().iloc[0]]
        else:
            row = df.iloc[(target - 70.0).abs().argsort().iloc[0]]
    return {name: float(row[name]) for name in DYNAMIC_INPUT_FEATURES}


DYNAMIC_DEFAULT_INPUT = dynamic_default_input()
TREND_CAPPING_RULES = load_trend_capping_rules(TREND_CAPPING_PATH)


def load_trend_reason_rules() -> dict[str, dict[str, str]]:
    if not TREND_REASONS_PATH.exists():
        return {}
    df = pd.read_excel(TREND_REASONS_PATH, sheet_name="Trend_HI_Reasons")
    reasons = {}
    for record in df.to_dict("records"):
        parameter = str(record.get("Parameter", "")).strip()
        if not parameter:
            continue
        reasons[parameter] = {
            "why": str(record.get("Why This Trend Affects HI", "")).strip(),
            "cause": str(record.get("Likely Technical Cause", "")).strip(),
            "message": str(record.get("Recommended User Message", "")).strip(),
            "action": str(record.get("Recommended Action", "")).strip(),
            "hi_effect": str(record.get("Expected HI Effect Before Hard Cap", "")).strip(),
        }
    return reasons


TREND_REASON_RULES = load_trend_reason_rules()


def classify_hi(value: float) -> str:
    if value >= 85:
        return "Healthy"
    if value >= 65:
        return "Moderate"
    if value >= 40:
        return "Poor"
    return "Critical"


def gas_issue_reason(gas: str, value: float) -> dict[str, str] | None:
    rule = CAPPING_REASON_RULES["gas"].get(gas)
    if not rule:
        return None
    if value_in_range(value, rule["critical_range"]):
        return {"parameter": gas, "value": f"{value:g} ppm", "condition": "Critical", "reason": rule["critical_reason"]}
    if value_in_range(value, rule["poor_range"]):
        return {"parameter": gas, "value": f"{value:g} ppm", "condition": "Poor", "reason": rule["poor_reason"]}
    return None


def temperature_issue_reason(parameter: str, value: float) -> dict[str, str] | None:
    rule = CAPPING_REASON_RULES["temperature"].get(parameter)
    if not rule:
        return None
    if parameter == "OTI" and value > 90:
        return {"parameter": "OTI(MAX)", "value": f"{value:g} °C", "condition": str(rule["Condition"]), "reason": str(rule["Reason"])}
    if parameter == "WTI" and value > 95:
        return {"parameter": "WTI(MAX)", "value": f"{value:g} °C", "condition": str(rule["Condition"]), "reason": str(rule["Reason"])}
    return None


def bdv_issue_reason(value: float) -> dict[str, str] | None:
    for rule in CAPPING_REASON_RULES["bdv"]:
        if value_in_range(value, rule["BDV (kV)"]):
            condition = str(rule["Condition"]).strip()
            if condition == "Good":
                return None
            return {"parameter": "BDV", "value": f"{value:g} kV", "condition": condition, "reason": str(rule["Reason"])}
    return None


def issue_reasons(row: dict[str, float], values: dict[str, float], gas_status: dict[str, str], operating_violations: dict[str, float]) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    for gas in RAW_FEATURES:
        reason = gas_issue_reason(gas, row[gas])
        if reason:
            reasons.append(reason)
    bdv_reason = bdv_issue_reason(float(values.get("BDV", DEFAULT_INPUT["BDV"])))
    if bdv_reason:
        reasons.append(bdv_reason)
    oti_reason = temperature_issue_reason("OTI", float(values.get("OTI", DEFAULT_INPUT["OTI"])))
    if oti_reason:
        reasons.append(oti_reason)
    wti_reason = temperature_issue_reason("WTI", float(values.get("WTI", max(DEFAULT_INPUT["WTI_HV"], DEFAULT_INPUT["WTI_LV"]))))
    if wti_reason:
        reasons.append(wti_reason)
    return reasons


def recommendation(condition: str, reasons: list[dict[str, str]]) -> str:
    if reasons:
        return "Review the issue points below."
    if condition == "Healthy":
        return "Transformer condition is healthy. Continue routine monitoring and periodic oil testing."
    if condition == "Moderate":
        return "Schedule closer monitoring. Review loading, cooling performance, and repeat DGA trend analysis."
    if condition == "Poor":
        return "Plan maintenance action. Verify oil quality, inspect cooling system, and increase testing frequency."
    return "Critical condition indicated. Carry out urgent diagnostic inspection and operational risk review."


def age_band_for(age_years: float) -> str:
    for label, low, high in AGE_BANDS:
        if label == "0–10 yr" and low <= age_years <= high:
            return label
        if low < age_years <= high:
            return label
    return ">40 yr"


def age_rule_limit(parameter_key: str, age_years: float) -> tuple[float | None, str, str]:
    rule = AGE_HARD_CAP_RULES.get(parameter_key)
    band = age_band_for(age_years)
    if not rule:
        return None, band, ""
    return rule["limits"].get(band), band, rule.get("reason", "")


def max_value(values: dict[str, float], keys: tuple[str, ...]) -> float:
    return max(float(values.get(key, 0.0)) for key in keys)


def min_value(values: dict[str, float], keys: tuple[str, ...]) -> float:
    return min(float(values.get(key, 0.0)) for key in keys)


def add_age_violation(
    violations: dict[str, dict[str, object]],
    key: str,
    value: float,
    limit: float | None,
    band: str,
    reason: str,
    label: str,
    unit: str,
    direction: str,
) -> None:
    if limit is None:
        return
    if direction == "max" and value < limit:
        return
    if direction == "min" and value >= limit:
        return
    operator = ">=" if direction == "max" else "<"
    violations[key] = {
        "value": value,
        "limit": limit,
        "band": band,
        "reason": reason,
        "label": label,
        "unit": unit,
        "operator": operator,
    }


def operating_limit_violations(values: dict[str, float]) -> dict[str, dict[str, object]]:
    values = add_calculated_inputs(values)
    violations: dict[str, dict[str, object]] = {}
    age_years = float(values.get("Age_Years", DEFAULT_INPUT["Age_Years"]))

    checks = (
        ("h2 ppm", "H2", float(values["H2"]), "H2", "ppm", "max"),
        ("ch4 ppm", "Methane", float(values["Methane"]), "Methane", "ppm", "max"),
        ("c2h6 ppm", "Ethane", float(values["Ethane"]), "Ethane", "ppm", "max"),
        ("c2h4 ppm", "Ethylene", float(values["Ethylene"]), "Ethylene", "ppm", "max"),
        ("c2h2 ppm", "Acetylene", float(values["Acetylene"]), "Acetylene", "ppm", "max"),
        ("co ppm*", "CO", float(values["CO"]), "CO", "ppm", "max"),
        ("co2 ppm*", "CO2", float(values["CO2"]), "CO2", "ppm", "max"),
        ("moisture ppm", "Moisture_PPM", float(values["Moisture_PPM"]), "Moisture", "ppm", "max"),
        ("hv/lv wti °c", "WTI", float(values["WTI"]), "WTI(MAX)", "deg C", "max"),
        ("oti °c", "OTI", float(values["OTI"]), "OTI", "deg C", "max"),
        ("loading %", "Loading_pct", float(values["Loading_pct"]), "Loading", "%", "max"),
        (
            "winding tan delta %",
            "Winding_TanDelta",
            max_value(values, ("Winding_HV_TanDelta_pct", "Winding_LV_TanDelta_pct", "Winding_TV_TanDelta_pct")),
            "Max Winding Tan Delta",
            "%",
            "max",
        ),
        (
            "bushing tan delta %",
            "Bushing_TanDelta",
            max_value(
                values,
                (
                    "Bushing_HV_R_TanDelta_pct",
                    "Bushing_HV_Y_TanDelta_pct",
                    "Bushing_HV_B_TanDelta_pct",
                    "Bushing_LV_R_TanDelta_pct",
                    "Bushing_LV_Y_TanDelta_pct",
                    "Bushing_LV_B_TanDelta_pct",
                    "Bushing_TV_R_TanDelta_pct",
                    "Bushing_TV_Y_TanDelta_pct",
                    "Bushing_TV_B_TanDelta_pct",
                ),
            ),
            "Max Bushing Tan Delta",
            "%",
            "max",
        ),
        (
            "capacitance change %",
            "Capacitance_Change",
            max(abs(float(values[key])) for key in INPUT_FEATURES if key.endswith("_CapChange_pct")),
            "Max Capacitance Change",
            "%",
            "max",
        ),
        ("bdv minimum kv", "BDV", float(values["BDV"]), "BDV", "kV", "min"),
        (
            "ir 1-min minimum mω",
            "IR_Min_1min",
            min_value(values, ("IR_HV_E_1min", "IR_LV_E_1min", "IR_TV_E_1min")),
            "IR Min 1 min",
            "Mohm",
            "min",
        ),
        (
            "ir 10-min minimum mω",
            "IR_Min_10min",
            min_value(values, ("IR_HV_E_10min", "IR_LV_E_10min", "IR_TV_E_10min")),
            "IR Min 10 min",
            "Mohm",
            "min",
        ),
        ("pi minimum", "PI_Min", min_value(values, ("PI_HV", "PI_LV", "PI_TV")), "PI Min", "", "min"),
    )

    for parameter_key, key, value, label, unit, direction in checks:
        limit, band, reason = age_rule_limit(parameter_key, age_years)
        add_age_violation(violations, key, value, limit, band, reason, label, unit, direction)

    if "cft ir minimum mω" in AGE_HARD_CAP_RULES and "CFT_IR_Min" in values:
        limit, band, reason = age_rule_limit("cft ir minimum mω", age_years)
        add_age_violation(violations, "CFT_IR_Min", float(values["CFT_IR_Min"]), limit, band, reason, "CFT IR Min", "Mohm", "min")
    return violations


def hard_cap_reasons(violations: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    reasons = []
    for item in violations.values():
        value = float(item["value"])
        limit = float(item["limit"])
        unit = str(item["unit"])
        band = str(item["band"])
        label = str(item["label"])
        reason = str(item["reason"]) or "The measured value is outside the acceptable age-based limit."
        direction = "maximum" if str(item["operator"]) == ">=" else "minimum"
        comparison = "above" if direction == "maximum" else "below"
        formatted_value = f"{value:g} {unit}".strip()
        limit_text = f"{limit:g} {unit}".strip()
        message = (
            f"For transformer age group {band}, {label} is {comparison} the {direction} acceptable limit "
            f"of {limit_text}. {reason} The transformer should be treated as Critical until this issue is "
            "verified and corrected."
        )
        reasons.append(
            {
                "parameter": label,
                "value": formatted_value,
                "condition": "Critical",
                "reason": message,
            }
        )
    return reasons


def predict_payload(values: dict[str, float]) -> dict:
    values = add_calculated_inputs(values)
    model = load_model()
    row = {name: float(values.get(name, DEFAULT_INPUT[name])) for name in RAW_FEATURES}
    row.update(
        {
            "BDV": float(values.get("BDV", DEFAULT_INPUT["BDV"])),
            "OTI": float(values.get("OTI", DEFAULT_INPUT["OTI"])),
            "WTI": float(values.get("WTI", max(DEFAULT_INPUT["WTI_HV"], DEFAULT_INPUT["WTI_LV"]))),
        }
    )
    features = np.array([[float(values[name]) for name in INPUT_FEATURES]], dtype=np.float64)
    model_hi = float(predict(model, features)[0][0])
    scores = gas_scores(row)
    summary = rule_summary(scores)
    gas_status = gas_conditions(row)
    label_features = label_rule_features(scores)
    label_rule = label_rule_name(scores)
    critical_label_rule = label_features["label_rule_any_gas_critical"] == 1.0
    operating_violations = operating_limit_violations(values)
    operating_override = bool(operating_violations)
    hi = 0.0 if operating_override else model_hi
    condition = classify_hi(hi)
    reasons = issue_reasons(row, values, gas_status, operating_violations)
    existing_reason_keys = {(item["parameter"], item["value"]) for item in reasons}
    for item in hard_cap_reasons(operating_violations):
        if (item["parameter"], item["value"]) not in existing_reason_keys:
            reasons.append(item)

    return {
        "transformer_details": {
            "input_count": len(INPUT_FEATURES),
        },
        "health_index": round(hi, 2),
        "model_health_index": round(model_hi, 2),
        "label_rule_override": bool(critical_label_rule),
        "operating_limit_override": bool(operating_override),
        "operating_violations": operating_violations,
        "condition": condition,
        "recommendation": recommendation(condition, reasons),
        "reason_points": reasons,
        "gas_conditions": gas_status,
        "gas_scores": {key: round(value, 2) for key, value in scores.items()},
        "operating_scores": {key: round(value, 2) for key, value in operating_scores(row).items()},
        "pi_values": {key: round(values[key], 3) for key in ("PI_HV", "PI_LV", "PI_TV")},
        "pi_conditions": {key: pi_condition(values[key]) for key in ("PI_HV", "PI_LV", "PI_TV")},
        "label_rule": label_rule,
        "label_rule_features": {key: round(value, 2) for key, value in label_features.items()},
        "rule_summary": {key: round(value, 2) for key, value in summary.items()},
    }


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def condition_class(condition: str) -> str:
    return condition if condition in {"Healthy", "Moderate", "Poor", "Critical"} else "Critical"


def bar_row(label: str, value: float, unit: str, scale: float, condition: str = "") -> str:
    width = max(2.0, min(100.0, (value / scale) * 100.0 if scale else 0.0))
    safe_label = html.escape(label)
    safe_condition = html.escape(condition)
    condition_badge = f"<span>{safe_condition}</span>" if safe_condition else ""
    return f"""
      <div class="bar-row">
        <div class="bar-head"><strong>{safe_label}</strong><em>{value:g} {html.escape(unit)}</em>{condition_badge}</div>
        <div class="bar-track"><i style="width:{width:.2f}%"></i></div>
      </div>
    """


def input_field_html(key: str) -> str:
    label, unit = INPUT_LABELS[key]
    suffix = f" ({unit})" if unit else ""
    readonly = ' readonly tabindex="-1"' if key in {"PI_HV", "PI_LV", "PI_TV"} else ""
    return (
        f'<label><span>{html.escape(label + suffix)}</span>'
        f'<input name="{html.escape(key)}" type="number" step="any" '
        f'value="{DEFAULT_INPUT[key]:g}"{readonly}></label>'
    )


def input_sections_html() -> str:
    sections = []
    for title, keys in INPUT_SECTIONS:
        fields = "\n".join(input_field_html(key) for key in keys)
        sections.append(
            f"""
        <section class="form-section">
          <h3>{html.escape(title)}</h3>
          <div class="fields">
            {fields}
          </div>
        </section>
            """
        )
    return "\n".join(sections)


def split_dynamic_feature(name: str) -> tuple[str, str]:
    for suffix in ("_Tminus2", "_Tminus1", "_T0_Current"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return name, ""


def dynamic_input_label(name: str) -> tuple[str, str]:
    base, _ = split_dynamic_feature(name)
    current_key = DYNAMIC_COLUMN_TO_CURRENT.get(base, base)
    label, unit = INPUT_LABELS.get(current_key, (base.replace("_", " "), ""))
    return label, unit


def dynamic_input_field_html(key: str) -> str:
    label, unit = dynamic_input_label(key)
    suffix = f" ({unit})" if unit else ""
    readonly = ' readonly tabindex="-1"' if any(key.startswith(f"PI_{phase}") for phase in ("HV_", "LV_", "TV_")) else ""
    return (
        f'<label><span>{html.escape(label + suffix)}</span>'
        f'<input name="{html.escape(key)}" type="number" step="any" '
        f'value="{DYNAMIC_DEFAULT_INPUT[key]:g}"{readonly}></label>'
    )


def dynamic_input_sections_html() -> str:
    sections = []
    for time_key, title in TIME_POINT_LABELS.items():
        keys = [key for key in DYNAMIC_INPUT_FEATURES if key.endswith(f"_{time_key}")]
        fields = "\n".join(dynamic_input_field_html(key) for key in keys)
        year_field = TIME_POINT_YEAR_FIELD[time_key]
        default_year = TIME_POINT_DEFAULT_YEAR[time_key]
        sections.append(
            f"""
        <section class="form-section">
          <div class="year-box">
            <h3>{html.escape(title)}</h3>
            <label><span>Year</span><input name="{html.escape(year_field)}" type="text" value="{html.escape(default_year)}"></label>
          </div>
          <div class="fields">
            {fields}
          </div>
        </section>
            """
        )
    return "\n".join(sections)


def render_result_page(query: dict[str, list[str]], raw_query: str = "") -> str:
    values = add_calculated_inputs(parse_prediction_values(query))
    substation_name = html.escape(query_text(query, "SubstationName", "Substation-1"))
    voltage_ratio = html.escape(query_text(query, "VoltageRatio", "220/132 kV"))
    pdf_fields = hidden_form_fields(query)
    data = predict_payload(values)
    gas_status = data["gas_conditions"]
    gas_scale = {rule.column: rule.poor_max for rule in health_rules.GAS_RULES}

    gas_bars = "\n".join(
        bar_row(label, values[column], "ppm", gas_scale.get(column, max(values[column], 1)), gas_status[column])
        for label, column in (
            ("H2 Hydrogen", "H2"),
            ("CH4 Methane", "Methane"),
            ("C2H6 Ethane", "Ethane"),
            ("C2H4 Ethylene", "Ethylene"),
            ("C2H2 Acetylene", "Acetylene"),
            ("CO Carbon Monoxide", "CO"),
            ("CO2 Carbon Dioxide", "CO2"),
        )
    )
    operating_bars = "\n".join(
        [
            bar_row("BDV", values["BDV"], "kV", 80),
            bar_row("OTI(MAX)", values["OTI"], "°C", 120),
            bar_row("WTI(MAX)", values["WTI"], "°C", 130),
            bar_row("Loading", values["Loading_pct"], "%", 150),
            bar_row("Ambient", values["Ambient_Temp"], "°C", 60),
            bar_row("Age", values["Age_Years"], "years", 50),
            bar_row("Moisture", values["Moisture_PPM"], "ppm", 60),
        ]
    )
    ir_bars = "\n".join(
        bar_row(INPUT_LABELS[column][0], values[column], INPUT_LABELS[column][1], max(values[column], 10000))
        for column in (
            "IR_HV_E_1min",
            "IR_LV_E_1min",
            "IR_TV_E_1min",
            "IR_HV_E_10min",
            "IR_LV_E_10min",
            "IR_TV_E_10min",
            "PI_HV",
            "PI_LV",
            "PI_TV",
        )
    )
    reason_cards = "\n".join(
        f"""
        <article class="reason-card">
          <div class="reason-top">
            <strong>{html.escape(str(item["parameter"]))}</strong>
            <span class="condition-pill">{html.escape(item["condition"])}</span>
          </div>
          <dl>
            <div><dt>Observed value</dt><dd>{html.escape(item["value"])}</dd></div>
            <div><dt>Recommendation</dt><dd>{html.escape(item["reason"])}</dd></div>
          </dl>
        </article>
        """
        for item in data.get("reason_points", [])
    )
    if not reason_cards:
        reason_cards = '<article class="reason-card ok"><div class="reason-top"><strong>No abnormal parameter found</strong><span class="condition-pill">Normal</span></div><p>All entered values are within the active rule limits. Continue routine monitoring and periodic testing as per schedule.</p></article>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transformer Health Result</title>
  <style>
    :root {{
      --bg: #03070c;
      --panel: rgba(9, 22, 34, .82);
      --line: rgba(119, 221, 255, .2);
      --text: #eefaff;
      --muted: #9bb2c0;
      --cyan: #2fe6ff;
      --green: #40ffa8;
      --amber: #ffd166;
      --red: #ff5d6c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 18%, rgba(47,230,255,.16), transparent 30%),
        radial-gradient(circle at 78% 72%, rgba(64,255,168,.12), transparent 32%),
        linear-gradient(135deg, #010306, #071421 52%, #020509);
      padding: clamp(16px, 3vw, 34px);
    }}
    .layout {{
      width: min(1180px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .layout > * {{
      animation: riseIn .62s ease both;
    }}
    .layout > *:nth-child(2) {{ animation-delay: .08s; }}
    .layout > *:nth-child(3) {{ animation-delay: .16s; }}
    .layout > *:nth-child(4) {{ animation-delay: .24s; }}
    .top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 22px;
      background: var(--panel);
      box-shadow: 0 20px 70px rgba(0,0,0,.35);
    }}
    .top-title {{
      display: grid;
      gap: 14px;
    }}
    .result-title-row {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .result-logo {{
      width: 76px;
      height: 76px;
      object-fit: contain;
      border-radius: 50%;
      background: rgba(255,255,255,.94);
      padding: 3px;
    }}
    .top-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .result-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      border: 1px solid rgba(47, 230, 255, .42);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--text);
      text-decoration: none;
      font-weight: 800;
      background: rgba(255, 255, 255, .07);
    }}
    .result-btn.primary {{
      color: #021019;
      background: linear-gradient(135deg, var(--cyan), var(--green));
      box-shadow: 0 0 24px rgba(47, 230, 255, .18);
    }}
    .result-form {{ margin: 0; }}
    .result-form button {{
      font: inherit;
      cursor: pointer;
    }}
    h1 {{ margin: 0; font-size: clamp(26px, 4vw, 44px); }}
    .sub {{ color: var(--muted); margin-top: 8px; }}
    .score {{
      text-align: right;
      min-width: 180px;
    }}
    .score strong {{
      display: block;
      font-size: clamp(46px, 8vw, 76px);
      line-height: .9;
      color: var(--green);
      text-shadow: 0 0 26px rgba(64,255,168,.26);
    }}
    .badge {{
      display: inline-block;
      margin-top: 10px;
      padding: 8px 13px;
      border-radius: 999px;
      font-weight: 800;
      color: #041018;
      background: var(--green);
    }}
    .badge.Moderate {{ background: var(--amber); }}
    .badge.Poor, .badge.Critical {{ color: white; background: var(--red); }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
      background: var(--panel);
    }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .detail-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}
    .detail {{
      border: 1px solid rgba(145,205,224,.16);
      border-radius: 8px;
      padding: 12px;
      background: rgba(2,8,13,.45);
    }}
    .detail small {{ display: block; color: var(--muted); margin-bottom: 5px; }}
    .detail strong {{ font-size: 17px; }}
    .bar-row {{ margin-bottom: 13px; }}
    .bar-head {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 7px;
      color: #dffbff;
      font-size: 13px;
    }}
    .bar-head em {{ color: var(--muted); font-style: normal; }}
    .bar-head span {{ color: var(--amber); }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }}
    .bar-track i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      box-shadow: 0 0 18px rgba(47,230,255,.25);
    }}
    .reason-card {{
      border: 1px solid rgba(255,93,108,.28);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
      background: rgba(255,93,108,.08);
    }}
    .reason-card.ok {{
      border-color: rgba(64,255,168,.25);
      background: rgba(64,255,168,.08);
    }}
    .reason-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .reason-card strong {{ color: #fff; font-size: 16px; }}
    .condition-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(255,220,160,.12);
      color: #ffdca0;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .reason-card dl {{
      display: grid;
      gap: 10px;
      margin: 0;
    }}
    .reason-card dl div {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .reason-card dt {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .reason-card dd {{
      margin: 0;
      color: #dcecf2;
      line-height: 1.55;
    }}
    .reason-card p {{ margin: 0; color: #dcecf2; line-height: 1.55; }}
    .recommendation-summary {{
      margin: 0 0 14px;
      padding: 13px 15px;
      border: 1px solid rgba(47,230,255,.18);
      border-radius: 8px;
      background: rgba(47,230,255,.06);
      color: #dffbff;
      line-height: 1.55;
    }}
    .meta {{ color: var(--muted); line-height: 1.5; }}
    .result-footer {{
      text-align: center;
      color: #9fd4df;
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
      padding: 8px 0 2px;
    }}
    .result-footer a, .hero-credit a, .modal-footer a {{
      color: #fff;
      text-decoration: none;
    }}
    .result-footer a:hover, .hero-credit a:hover, .modal-footer a:hover {{
      color: #fff;
    }}
    @keyframes riseIn {{
      from {{ opacity: 0; transform: translateY(18px) scale(.985); filter: blur(5px); }}
      to {{ opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }}
    }}
    @media (max-width: 720px) {{
      .reason-top {{ align-items: flex-start; flex-direction: column; }}
      .reason-card dl div {{ grid-template-columns: 1fr; gap: 4px; }}
    }}
    @media (max-width: 860px) {{
      .top, .grid, .detail-row {{ grid-template-columns: 1fr; display: grid; }}
      .score {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <section class="top">
      <div class="top-title">
        <div class="result-title-row"><img class="result-logo" src="{LOGO_URL}" alt="Power Asset Intelligence logo"><h1>Transformer Health Result</h1></div>
        <div class="sub">AI-based Health Index Prediction for Transformers</div>
        <div class="top-actions">
          <a class="result-btn" href="/">Home Page</a>
          <a class="result-btn" href="/?input=1">Back to Input</a>
          <a class="result-btn" href="/contact">Contact Us</a>
          <form class="result-form" method="post" action="/result.pdf">
            {pdf_fields}
            <button class="result-btn primary" type="submit">Download PDF</button>
          </form>
        </div>
      </div>
      <div class="score">
        <strong>{data["health_index"]:.2f}</strong>
        <span class="badge {condition_class(data["condition"])}">{html.escape(data["condition"])}</span>
      </div>
    </section>

    <section class="panel">
      <h2>Transformer Details</h2>
      <div class="detail-row">
        <div class="detail"><small>Substation</small><strong>{substation_name}</strong></div>
        <div class="detail"><small>Model Inputs</small><strong>{len(INPUT_FEATURES)} parameters</strong></div>
        <div class="detail"><small>Voltage Ratio</small><strong>{voltage_ratio}</strong></div>
        <div class="detail"><small>Model Estimate</small><strong>{data["model_health_index"]:.2f}</strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>DGA Input Graph</h2>
        {gas_bars}
      </div>
      <div class="panel">
        <h2>Oil, Thermal &amp; Age Graph</h2>
        {operating_bars}
      </div>
    </section>

    <section class="panel">
      <h2>IR / CFT Input Graph</h2>
      {ir_bars}
    </section>

    <section class="panel">
      <h2>Recommendation</h2>
      <p class="recommendation-summary">{html.escape(data["recommendation"])}</p>
      {reason_cards}
    </section>

    <footer class="result-footer"><a href="/contact">Created by Satya</a></footer>
  </main>
</body>
</html>"""


def report_id_from_datetime(value: datetime) -> str:
    return f"PAI-THM-{value.strftime('%S%M')}"


def apply_cover_template(
    content_pdf: bytes,
    *,
    report_type: str,
    username: str,
    organisation: str,
    substation_name: str,
    voltage_ratio: str,
    timestamp: str,
    report_id: str,
) -> bytes:
    if not COVER_TEMPLATE_PATH.exists():
        return content_pdf
    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
    except Exception:
        return content_pdf

    template_reader = PdfReader(str(COVER_TEMPLATE_PATH))
    content_reader = PdfReader(io.BytesIO(content_pdf))
    cover_page = template_reader.pages[0]
    width = float(cover_page.mediabox.width)
    height = float(cover_page.mediabox.height)

    overlay_buffer = io.BytesIO()
    c = canvas.Canvas(overlay_buffer, pagesize=(width, height))
    navy = colors.HexColor("#082b5f")
    body = colors.HexColor("#222222")
    green = colors.HexColor("#19864b")

    # Clean only the sample text areas while preserving the supplied cover artwork.
    c.setFillColor(colors.white)
    c.rect(100, 121, 202, 143, fill=1, stroke=0)
    c.rect(306, 121, 208, 143, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.rect(500, 116, 24, 126, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#c9cfd6"))
    c.setLineWidth(0.55)
    c.line(306, 126, 306, 264)
    icon_specs = [
        (COVER_ICON_TRANSFORMER_PATH, 314, 232),
        (COVER_ICON_DATE_PATH, 314, 178),
        (COVER_ICON_REPORT_ID_PATH, 314, 124),
    ]
    for icon_path, icon_x, icon_y in icon_specs:
        if icon_path.exists():
            c.drawImage(str(icon_path), icon_x, icon_y, width=32, height=32, preserveAspectRatio=True, mask="auto")
    def draw_pair(x: float, y: float, label: str, value: str, max_chars: int = 34) -> None:
        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 8.2)
        c.drawString(x, y, label)
        c.setFillColor(body)
        c.setFont("Helvetica", 7.8)
        clean = (value or "-").strip() or "-"
        if len(clean) > max_chars:
            clean = clean[: max_chars - 3].rstrip() + "..."
        c.drawString(x, y - 11, clean)

    report_type_labels = {
        "Current": "Current Health Index",
        "Trend": "Trend-Based Health Index",
    }
    display_report_type = report_type_labels.get(report_type, report_type)
    draw_pair(108, 245, "REPORT TYPE", display_report_type, 40)
    draw_pair(108, 211, "USER NAME", username, 40)
    draw_pair(108, 177, "ORGANISATION", organisation, 40)
    draw_pair(108, 143, "SUBSTATION", substation_name, 40)
    draw_pair(368, 245, "TRANSFORMER", voltage_ratio, 31)
    draw_pair(368, 191, "REPORT DATE", timestamp, 31)
    draw_pair(368, 137, "REPORT ID", report_id, 31)

    c.setFillColor(green)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawRightString(width - 36, 25, timestamp)
    c.save()

    overlay_buffer.seek(0)
    overlay_page = PdfReader(overlay_buffer).pages[0]
    cover_page.merge_page(overlay_page)

    writer = PdfWriter()
    writer.add_page(cover_page)
    for page in content_reader.pages:
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def render_result_pdf(query: dict[str, list[str]]) -> tuple[bytes, str]:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    values = add_calculated_inputs(parse_prediction_values(query))
    data = predict_payload(values)
    substation_name = query_text(query, "SubstationName", "Substation-1")
    username = query_text(query, "Username", "")
    designation = query_text(query, "Designation", "")
    organisation = query_text(query, "Organisation", "")
    voltage_ratio = query_text(query, "VoltageRatio", "220/132 kV")
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    report_id = report_id_from_datetime(generated_at)
    filename = f"health_index_{safe_filename_part(substation_name)}_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"

    buffer = io.BytesIO()
    site_url = public_base_url()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=58, bottomMargin=48)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#12313d")
    styles["Heading1"].textColor = colors.HexColor("#0f766e")
    styles["Heading2"].textColor = colors.HexColor("#12313d")
    title_style = ParagraphStyle("CenteredTitle", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica", fontSize=18, leading=24, textColor=colors.HexColor("#12313d"))
    heading_style = ParagraphStyle("CenteredHeading", parent=styles["Heading2"], alignment=TA_CENTER, fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor("#12313d"), spaceAfter=10)

    def draw_header_footer(canvas, document):
        width, height = A4
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 42)
        canvas.setFillColor(colors.HexColor("#12313d"))
        try:
            canvas.setFillAlpha(0.045)
        except AttributeError:
            pass
        canvas.translate(width / 2, height / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "POWER ASSET INTELLIGENCE")
        canvas.restoreState()
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#12313d"))
        canvas.rect(0, height - 42, width, 42, fill=1, stroke=0)
        if LOGO_PATH.exists():
            canvas.drawImage(str(LOGO_PATH), 36, height - 40, width=34, height=34, preserveAspectRatio=True, mask="auto")
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(78, height - 26, "Transformer Health Monitoring System")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(width - 36, height - 26, "Current Health Index Report")
        canvas.setStrokeColor(colors.HexColor("#d5e7ed"))
        canvas.line(36, 34, width - 36, 34)
        canvas.setFillColor(colors.HexColor("#0f766e"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 20, site_url)
        canvas.linkURL(site_url, (36, 17, 132, 29), relative=0)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(width / 2, 20, f"Generated: {timestamp}")
        canvas.drawRightString(width - 36, 20, f"Page {document.page}")
        canvas.restoreState()

    story = [
        Paragraph("<u>Transformer Health Monitoring System</u>", title_style),
        Paragraph("<u>Current Health Index Result</u>", heading_style),
        Spacer(1, 10),
    ]

    summary_table = Table(
        [
            ["Generated", html.escape(timestamp)],
            ["Substation Name", html.escape(substation_name)],
            ["Username", html.escape(username) or "-"],
            ["Designation", html.escape(designation) or "-"],
            ["Organisation", html.escape(organisation) or "-"],
            ["Voltage Ratio", html.escape(voltage_ratio)],
            ["Model Inputs", f"{len(INPUT_FEATURES)} parameters"],
            ["Health Index", f"{data['health_index']:.2f}"],
            ["Condition", data["condition"]],
            ["Model Estimate", f"{data['model_health_index']:.2f}"],
            ["Label Rule", data["label_rule"]],
        ],
        colWidths=[150, 320],
    )
    summary_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fbfc")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8dce3")), ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e5ea")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 8)]))
    story.extend([summary_table, Spacer(1, 14), Paragraph("<u>Input Parameters</u>", heading_style)])

    input_rows = [["Parameter", "Value"]]
    severity_rows: dict[int, str] = {}
    gas_status = data.get("gas_conditions", {})
    violation_keys = set(data.get("operating_violations", {}).keys())
    critical_labels = {str(item["parameter"]).lower() for item in data.get("reason_points", []) if str(item.get("condition", "")).lower() == "critical"}
    warning_labels = {str(item["parameter"]).lower() for item in data.get("reason_points", []) if str(item.get("condition", "")).lower() in {"poor", "poor/critical"}}
    for column in INPUT_FEATURES:
        label, unit = INPUT_LABELS[column]
        row_index = len(input_rows)
        input_rows.append([label, f"{values[column]:g} {unit}"])
        label_key = label.lower()
        if column in violation_keys or label_key in critical_labels or any(label_key.startswith(item) or item in label_key for item in critical_labels):
            severity_rows[row_index] = "critical"
        elif gas_status.get(column) == "Critical":
            severity_rows[row_index] = "critical"
        elif gas_status.get(column) == "Poor" or label_key in warning_labels or any(label_key.startswith(item) or item in label_key for item in warning_labels):
            severity_rows[row_index] = "warning"
    input_table = Table(input_rows, colWidths=[220, 250])
    table_commands = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9edf7")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8cbd1")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 5), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    for row_index, severity in severity_rows.items():
        table_commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f8d7da" if severity == "critical" else "#fff3cd")))
    input_table.setStyle(TableStyle(table_commands))
    story.extend([input_table, PageBreak(), Paragraph("<u>Recommendation</u>", heading_style)])

    reason_points = data.get("reason_points", [])
    if reason_points:
        for item in reason_points:
            story.append(
                Paragraph(
                    f"<b>{html.escape(item['parameter'])}: {html.escape(item['value'])}</b><br/>"
                    f"Condition: {html.escape(item['condition'])}<br/>"
                    f"{html.escape(item['reason'])}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No abnormal parameter found. All entered values are within the active rule limits.", styles["BodyText"]))

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    pdf_body = apply_cover_template(
        buffer.getvalue(),
        report_type="Current",
        username=username,
        organisation=organisation,
        substation_name=substation_name,
        voltage_ratio=voltage_ratio,
        timestamp=timestamp,
        report_id=report_id,
    )
    return pdf_body, filename


def render_index() -> str:
    input_sections = input_sections_html()
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transformer Health Monitoring System</title>
  <style>
    :root {
      --bg: #03070c;
      --panel: rgba(9, 22, 34, .72);
      --panel-strong: rgba(11, 29, 44, .88);
      --line: rgba(119, 221, 255, .2);
      --text: #eefaff;
      --muted: #92a8b7;
      --cyan: #2fe6ff;
      --green: #40ffa8;
      --amber: #ffd166;
      --red: #ff5d6c;
      --shadow: rgba(47, 230, 255, .22);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 18%, rgba(47, 230, 255, .2), transparent 32%),
        radial-gradient(circle at 80% 70%, rgba(64, 255, 168, .16), transparent 30%),
        linear-gradient(135deg, #010306 0%, #06121e 48%, #020509 100%);
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .55;
      background-image:
        linear-gradient(rgba(47, 230, 255, .12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(47, 230, 255, .12) 1px, transparent 1px);
      background-size: 72px 72px;
      animation: gridMove 18s linear infinite;
      mask-image: linear-gradient(to bottom, rgba(0, 0, 0, .95), rgba(0, 0, 0, .08));
    }

    .pulse-line {
      position: fixed;
      left: -20%;
      width: 140%;
      height: 1px;
      pointer-events: none;
      background: linear-gradient(90deg, transparent, rgba(47, 230, 255, .84), rgba(64, 255, 168, .6), transparent);
      box-shadow: 0 0 22px var(--shadow);
      animation: pulseLine 6s ease-in-out infinite;
    }

    .pulse-line.one { top: 27%; }
    .pulse-line.two { top: 68%; animation-delay: -2.5s; opacity: .68; }

    .particles {
      position: fixed;
      inset: 0;
      overflow: hidden;
      pointer-events: none;
    }

    .particle {
      position: absolute;
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 14px var(--cyan);
      opacity: .45;
      animation: floatParticle 10s ease-in-out infinite;
    }

    .particle:nth-child(1) { left: 12%; top: 72%; animation-delay: -1s; }
    .particle:nth-child(2) { left: 26%; top: 22%; animation-delay: -4s; }
    .particle:nth-child(3) { left: 52%; top: 78%; animation-delay: -6s; }
    .particle:nth-child(4) { left: 74%; top: 28%; animation-delay: -2s; }
    .particle:nth-child(5) { left: 88%; top: 62%; animation-delay: -8s; }

    .shell {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      padding: 28px clamp(18px, 4vw, 56px);
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border: 1px solid rgba(47, 230, 255, .5);
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--green);
      box-shadow: 0 0 24px rgba(47, 230, 255, .18);
    }

    .brand-logo {
      width: 72px;
      height: 72px;
      object-fit: contain;
      border-radius: 50%;
      background: rgba(255, 255, 255, .94);
      padding: 3px;
      box-shadow: 0 0 24px rgba(47, 230, 255, .18);
    }

    .system-chip {
      border: 1px solid rgba(64, 255, 168, .28);
      border-radius: 999px;
      padding: 9px 14px;
      color: #c9ffed;
      background: rgba(64, 255, 168, .08);
      font-size: 13px;
    }

    .hero {
      display: grid;
      place-items: center;
      text-align: center;
      padding: 58px 0 36px;
    }

    .hero-inner {
      width: min(980px, 100%);
    }

    h1 {
      margin: 0;
      font-size: clamp(38px, 7vw, 82px);
      line-height: 1.02;
      letter-spacing: 0;
      text-shadow: 0 0 34px rgba(47, 230, 255, .24);
    }

    .subtitle {
      margin: 22px auto 0;
      max-width: 780px;
      color: #bdd4df;
      font-size: clamp(16px, 2.2vw, 23px);
      line-height: 1.5;
    }

    .hero-credit {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-top: 16px;
      border: 1px solid rgba(64, 255, 168, .3);
      border-radius: 999px;
      padding: 8px 16px;
      color: #d9fff2;
      background: rgba(64, 255, 168, .08);
      box-shadow: 0 0 24px rgba(47, 230, 255, .12);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .status-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(150px, 1fr));
      gap: 14px;
      margin: 42px auto 0;
      max-width: 760px;
    }

    .status-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: rgba(9, 22, 34, .55);
      backdrop-filter: blur(18px);
      box-shadow: 0 16px 44px rgba(0, 0, 0, .24);
    }

    .status-card strong {
      display: block;
      margin-bottom: 6px;
      color: var(--green);
      font-size: 22px;
    }

    .status-card span {
      color: var(--muted);
      font-size: 13px;
    }

    .cta-wrap {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: center;
      padding: 24px 0 18px;
    }

    .credit {
      text-align: center;
      color: #9fd4df;
      font-size: 14px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }

    .hero-credit a, .modal-footer a {
      color: #fff;
      text-decoration: none;
    }

    .hero-credit a:hover, .modal-footer a:hover {
      color: #fff;
    }

    .insert-btn, .trend-btn, .predict-btn, .home-btn, .reset-btn {
      border: 1px solid rgba(47, 230, 255, .62);
      border-radius: 8px;
      padding: 15px 24px;
      color: #021019;
      background: linear-gradient(135deg, var(--cyan), var(--green));
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, .6);
      transition: transform .2s ease, box-shadow .2s ease;
    }

    .trend-btn, .home-btn, .reset-btn {
      color: var(--text);
      background: rgba(255, 255, 255, .07);
      box-shadow: none;
    }

    .insert-btn:hover, .trend-btn:hover, .predict-btn:hover, .home-btn:hover, .reset-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 0 40px rgba(64, 255, 168, .28);
    }

    .modal {
      position: fixed;
      inset: 0;
      z-index: 5;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(0, 0, 0, .72);
      backdrop-filter: blur(14px);
    }

    .modal.open { display: flex; }

    .modal-card {
      width: min(1080px, 100%);
      max-height: 92vh;
      overflow: auto;
      border: 1px solid rgba(47, 230, 255, .28);
      border-radius: 10px;
      background: linear-gradient(145deg, rgba(9, 20, 32, .92), rgba(6, 14, 22, .96));
      box-shadow: 0 30px 90px rgba(0, 0, 0, .6), 0 0 40px rgba(47, 230, 255, .12);
    }

    .modal-head {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      padding: 20px 22px;
      background: rgba(6, 14, 22, .96);
      border-bottom: 1px solid rgba(47, 230, 255, .18);
    }

    .modal-head h2 { margin: 0; font-size: 22px; }

    .modal-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .close-btn {
      width: 38px;
      height: 38px;
      border: 1px solid rgba(255, 255, 255, .14);
      border-radius: 8px;
      color: var(--text);
      background: rgba(255, 255, 255, .06);
      cursor: pointer;
      font-size: 22px;
    }

    form {
      display: grid;
      gap: 18px;
      padding: 22px;
    }

    .form-section, .output-card {
      border: 1px solid rgba(47, 230, 255, .18);
      border-radius: 8px;
      padding: 18px;
      background: var(--panel);
      backdrop-filter: blur(18px);
    }

    .form-section h3, .output-card h3 {
      margin: 0 0 14px;
      color: #dffbff;
      font-size: 16px;
    }

    .year-box {
      display: grid;
      justify-items: center;
      gap: 12px;
      margin-bottom: 22px;
      padding: 18px;
      border: 1px solid rgba(47, 230, 255, .24);
      border-radius: 10px;
      background: rgba(47, 230, 255, .06);
    }

    .year-box h3 {
      margin: 0;
      font-size: 26px;
      text-align: center;
    }

    .year-box label {
      width: min(260px, 100%);
      text-align: center;
    }

    .year-box label span {
      font-size: 14px;
    }

    .year-box input {
      text-align: center;
      font-size: 26px;
      font-weight: 900;
    }

    .fields {
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 14px;
    }


    label span {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 13px;
    }

    input {
      width: 100%;
      border: 1px solid rgba(145, 205, 224, .22);
      border-radius: 7px;
      padding: 12px;
      color: var(--text);
      background: rgba(2, 8, 13, .62);
      font: inherit;
      outline: none;
    }

    input:focus {
      border-color: rgba(47, 230, 255, .75);
      box-shadow: 0 0 0 3px rgba(47, 230, 255, .12);
    }

    input[readonly] {
      color: #c9ffed;
      border-color: rgba(64, 255, 168, .26);
      background: rgba(64, 255, 168, .08);
      cursor: default;
    }

    .actions {
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }

    .actions .predict-btn {
      min-width: min(360px, 100%);
      padding: 18px 34px;
      font-size: 17px;
      letter-spacing: .02em;
    }

    .modal-footer {
      border-top: 1px solid rgba(47, 230, 255, .16);
      padding: 15px 22px 18px;
      text-align: center;
      color: #9fd4df;
      font-size: 13px;
      letter-spacing: .12em;
      text-transform: uppercase;
      background: rgba(6, 14, 22, .72);
    }

    .loading-overlay {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: none;
      place-items: center;
      background:
        linear-gradient(rgba(3, 7, 12, .88), rgba(3, 7, 12, .94)),
        repeating-linear-gradient(90deg, rgba(47,230,255,.08) 0 1px, transparent 1px 110px),
        repeating-linear-gradient(0deg, rgba(64,255,168,.06) 0 1px, transparent 1px 110px);
      backdrop-filter: blur(10px);
    }

    .loading-overlay.show { display: grid; }

    .loader-card {
      width: min(620px, calc(100vw - 36px));
      border: 1px solid rgba(47, 230, 255, .34);
      border-radius: 12px;
      padding: 30px;
      text-align: center;
      background: rgba(7, 18, 28, .92);
      box-shadow: 0 0 60px rgba(47,230,255,.18), inset 0 0 40px rgba(64,255,168,.05);
      position: relative;
      overflow: hidden;
    }

    .loader-card::before {
      content: "";
      position: absolute;
      left: -30%;
      right: -30%;
      top: 50%;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--cyan), var(--green), transparent);
      animation: scanLine 1.4s ease-in-out infinite;
      opacity: .85;
    }

    .loader-ring, .tower-mark {
      width: 86px;
      height: 86px;
      margin: 0 auto 18px;
      border-radius: 50%;
      border: 2px solid rgba(47,230,255,.6);
      display: grid;
      place-items: center;
      color: var(--green);
      font-weight: 900;
      font-size: 28px;
      animation: gridPulse 1.2s ease-in-out infinite;
      box-shadow: 0 0 34px rgba(47,230,255,.24);
    }

    .loader-card strong {
      display: block;
      margin-bottom: 10px;
      font-size: clamp(24px, 4vw, 34px);
      color: #eaffff;
    }

    .loader-card span {
      color: var(--muted);
      line-height: 1.55;
    }

    .power-steps {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 22px;
    }

    .power-steps i {
      display: block;
      min-height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
      position: relative;
    }

    .power-steps i::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      transform: translateX(-100%);
      animation: chargeStep 1.5s ease-in-out infinite;
    }

    .power-steps i:nth-child(2)::after { animation-delay: .18s; }
    .power-steps i:nth-child(3)::after { animation-delay: .36s; }

    @keyframes gridPulse {
      0%, 100% { transform: scale(1); box-shadow: 0 0 28px rgba(47,230,255,.18); }
      50% { transform: scale(1.06); box-shadow: 0 0 48px rgba(64,255,168,.3); }
    }

    @keyframes scanLine {
      0%, 100% { transform: translateY(-38px); opacity: .35; }
      50% { transform: translateY(42px); opacity: 1; }
    }

    @keyframes chargeStep {
      0% { transform: translateX(-100%); }
      55%, 100% { transform: translateX(0); }
    }

    .output-card {
      display: none;
      grid-template-columns: .75fr 1.25fr;
      gap: 18px;
      align-items: center;
    }

    .output-card.show { display: grid; }

    .score {
      font-size: clamp(46px, 8vw, 76px);
      font-weight: 900;
      color: var(--green);
      text-shadow: 0 0 26px rgba(64, 255, 168, .28);
    }

    .condition {
      display: inline-block;
      margin-top: 8px;
      border-radius: 999px;
      padding: 8px 13px;
      color: #041018;
      background: var(--green);
      font-weight: 800;
    }

    .condition.Moderate { background: var(--amber); }
    .condition.Poor, .condition.Critical { color: white; background: var(--red); }

    .recommendation {
      color: #d6e8ef;
      line-height: 1.55;
    }

    .recommendation-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }

    .recommendation-item {
      border: 1px solid rgba(47, 230, 255, .18);
      border-radius: 8px;
      padding: 12px;
      background: rgba(2, 8, 13, .45);
    }

    .recommendation-item strong {
      display: block;
      color: var(--cyan);
      margin-bottom: 5px;
    }

    .recommendation-item span {
      display: block;
      color: #ffdca0;
      font-size: 13px;
      margin-bottom: 6px;
    }

    .recommendation-item p {
      margin: 0;
      color: #d6e8ef;
      line-height: 1.45;
    }

    .meta {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    @keyframes gridMove {
      from { transform: translate3d(0, 0, 0); }
      to { transform: translate3d(72px, 72px, 0); }
    }

    @keyframes pulseLine {
      0%, 100% { transform: translateX(-18%) scaleX(.78); opacity: .15; }
      45% { transform: translateX(10%) scaleX(1); opacity: .9; }
    }

    @keyframes floatParticle {
      0%, 100% { transform: translateY(0) translateX(0); }
      50% { transform: translateY(-46px) translateX(18px); }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 860px) {
      .shell { padding: 20px 14px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .status-row, .fields, .output-card { grid-template-columns: 1fr; }
      .modal-card { max-height: 94vh; }
      .actions { justify-content: stretch; }
      .predict-btn { width: 100%; }
      .actions .predict-btn { min-width: 100%; }
      .modal-actions .home-btn { padding: 11px 13px; }
    }
  </style>
</head>
<body>
  <div class="pulse-line one"></div>
  <div class="pulse-line two"></div>
  <div class="particles"><i class="particle"></i><i class="particle"></i><i class="particle"></i><i class="particle"></i><i class="particle"></i></div>

  <main class="shell">
    <header class="topbar">
      <div class="brand"><img class="brand-logo" src="{LOGO_URL}" alt="Power Asset Intelligence logo"><span>Power Asset Intelligence</span></div>
      <div class="system-chip">Live AI Health Index Console</div>
    </header>

    <section class="hero">
      <div class="hero-inner">
        <h1>Transformer Health Monitoring System</h1>
        <p class="subtitle">AI-based Health Index Prediction for Transformers</p>
        <div class="hero-credit"><a href="/contact">Created by Satya</a></div>
        <div class="status-row" aria-label="system highlights">
          <div class="status-card"><strong>__INPUT_COUNT__</strong><span>Weighted dataset inputs</span></div>
          <div class="status-card"><strong>DGA + Oil</strong><span>Gas, BDV and moisture</span></div>
          <div class="status-card"><strong>IR / TD / Cap</strong><span>Insulation, winding and bushing signals</span></div>
        </div>
      </div>
    </section>

    <div class="cta-wrap">
      <button class="insert-btn" id="openModal" type="button">Current Health Index</button>
      <button class="trend-btn" id="trendHealthBtn" type="button">Trend-Based Health Index</button>
      <button class="trend-btn" type="button" onclick="window.location.href='/contact'">Contact Us</button>
    </div>
  </main>

  <section class="modal" id="dataModal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
      <div class="modal-head">
        <h2 id="modalTitle">Transformer Health Input</h2>
        <div class="modal-actions">
          <button class="home-btn" id="mainPageBtn" type="button">Main Page</button>
          <button class="reset-btn" type="button">Reset</button>
          <button class="close-btn" id="closeModal" type="button" aria-label="Close">x</button>
        </div>
      </div>
      <form id="healthForm" method="post" action="/result">
        <section class="form-section">
          <h3>Transformer Details</h3>
          <div class="fields">
            <label><span>Substation Name</span><input name="SubstationName" type="text" value="Substation-1"></label>
            <label><span>Username</span><input name="Username" type="text" value=""></label>
            <label><span>Designation</span><input name="Designation" type="text" value=""></label>
            <label><span>Organisation</span><input name="Organisation" type="text" value=""></label>
            <label><span>Voltage Ratio (kV)</span><input name="VoltageRatio" type="text" value="220/132 kV"></label>
            <label><span>Age (years)</span><input name="Age_Years" type="number" step="any" value="10.8"></label>
          </div>
        </section>

        __INPUT_SECTIONS__

        <div class="actions">
          <button class="reset-btn" type="button">Reset</button>
          <button class="predict-btn" type="submit">Predict Health Index</button>
        </div>
      </form>
      <footer class="modal-footer"><a href="/contact">Created by Satya</a></footer>
    </div>
  </section>

  <section class="loading-overlay" id="loadingOverlay" aria-hidden="true">
    <div class="loader-card">
      <div class="tower-mark">HV</div>
      <strong>Analyzing Transformer Health</strong>
      <span>Scanning DGA, insulation, thermal and oil-quality signals through the asset intelligence engine...</span>
      <div class="power-steps"><i></i><i></i><i></i></div>
    </div>
  </section>

  <script>
    const modal = document.getElementById('dataModal');
    const form = document.getElementById('healthForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const piFields = {
      PI_HV: form.elements['PI_HV'],
      PI_LV: form.elements['PI_LV'],
      PI_TV: form.elements['PI_TV']
    };

    function numericValue(name) {
      const field = form.elements[name];
      const value = Number.parseFloat(field ? field.value : '');
      return Number.isFinite(value) ? value : 0;
    }

    function calculatePi(tenMinuteName, oneMinuteName) {
      const tenMinute = numericValue(tenMinuteName);
      const oneMinute = numericValue(oneMinuteName);
      if (tenMinute <= 0 || oneMinute <= 0) return 0;
      return tenMinute / oneMinute;
    }

    function updatePiFields() {
      const values = {
        PI_HV: calculatePi('IR_HV_E_10min', 'IR_HV_E_1min'),
        PI_LV: calculatePi('IR_LV_E_10min', 'IR_LV_E_1min'),
        PI_TV: calculatePi('IR_TV_E_10min', 'IR_TV_E_1min')
      };
      Object.entries(values).forEach(([key, value]) => {
        if (piFields[key]) piFields[key].value = value > 0 ? value.toFixed(3) : '0.000';
      });
    }

    function clearFormFields() {
      Array.from(form.elements).forEach(field => {
        if (!field.name && field.type !== 'button') return;
        if (['button', 'submit'].includes(field.type)) return;
        if (field.type === 'checkbox' || field.type === 'radio') {
          field.checked = false;
          return;
        }
        field.value = '';
      });
    }

    [
      'IR_HV_E_1min',
      'IR_LV_E_1min',
      'IR_TV_E_1min',
      'IR_HV_E_10min',
      'IR_LV_E_10min',
      'IR_TV_E_10min'
    ].forEach(name => {
      if (form.elements[name]) form.elements[name].addEventListener('input', updatePiFields);
    });
    updatePiFields();

    document.getElementById('openModal').addEventListener('click', () => {
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    });

    document.getElementById('trendHealthBtn').addEventListener('click', () => {
      window.location.href = '/trend';
    });

    if (new URLSearchParams(window.location.search).get('input') === '1') {
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    }

    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('mainPageBtn').addEventListener('click', closeModal);
    document.querySelectorAll('.reset-btn').forEach(button => {
      button.addEventListener('click', clearFormFields);
    });
    modal.addEventListener('click', event => {
      if (event.target === modal) closeModal();
    });

    function closeModal() {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
    }

    form.addEventListener('submit', event => {
      event.preventDefault();
      loadingOverlay.classList.add('show');
      loadingOverlay.setAttribute('aria-hidden', 'false');
      setTimeout(() => form.submit(), 1800);
    });
  </script>
</body>
</html>"""
    return page.replace("__INPUT_COUNT__", str(len(INPUT_FEATURES))).replace("__INPUT_SECTIONS__", input_sections).replace("{LOGO_URL}", LOGO_URL)


def parse_dynamic_values(query: dict[str, list[str]]) -> dict[str, float]:
    return {key: float(query.get(key, [DYNAMIC_DEFAULT_INPUT[key]])[0]) for key in DYNAMIC_INPUT_FEATURES}


def trend_year_labels(query: dict[str, list[str]]) -> dict[str, str]:
    return {
        time_key: html.escape(str(query.get(field, [TIME_POINT_DEFAULT_YEAR[time_key]])[0]).strip() or TIME_POINT_DEFAULT_YEAR[time_key])
        for time_key, field in TIME_POINT_YEAR_FIELD.items()
    }


def dynamic_current_values(values: dict[str, float]) -> dict[str, float]:
    current_values: dict[str, float] = {}
    for key in INPUT_FEATURES:
        dataset_key = next((source for source, target in DYNAMIC_COLUMN_TO_CURRENT.items() if target == key), key)
        dynamic_key = f"{dataset_key}_T0_Current"
        if dynamic_key in values:
            current_values[key] = float(values[dynamic_key])
    return add_calculated_inputs(current_values)


def dynamic_values_for_time(values: dict[str, float], time_key: str) -> dict[str, float]:
    time_values: dict[str, float] = {}
    for key in INPUT_FEATURES:
        dataset_key = next((source for source, target in DYNAMIC_COLUMN_TO_CURRENT.items() if target == key), key)
        dynamic_key = f"{dataset_key}_{time_key}"
        if dynamic_key in values:
            time_values[key] = float(values[dynamic_key])
    return add_calculated_inputs(time_values)


def trend_change_points(values: dict[str, float]) -> list[dict[str, object]]:
    points = []
    for rule in TREND_CAPPING_RULES:
        parameter = str(rule["parameter"])
        old_key = f"{parameter}_Tminus2"
        previous_key = f"{parameter}_Tminus1"
        current_key = f"{parameter}_T0_Current"
        if old_key not in values or previous_key not in values or current_key not in values:
            continue
        old = float(values[old_key])
        previous = float(values[previous_key])
        current = float(values[current_key])
        bad_direction = str(rule["bad_direction"])
        floor = float(rule["denominator_floor"])
        if "absolute" in bad_direction:
            old_risk, previous_risk, current_risk = abs(old), abs(previous), abs(current)
            latest_delta = current_risk - previous_risk
            total_delta = current_risk - old_risk
        elif "lower" in bad_direction:
            latest_delta = previous - current
            total_delta = old - current
        else:
            latest_delta = current - previous
            total_delta = current - old
        latest_pct = (latest_delta / max(abs(previous), floor, 1e-9)) * 100.0
        total_pct = (total_delta / max(abs(old), floor, 1e-9)) * 100.0
        if latest_pct <= 0 and total_pct <= 0:
            continue
        points.append(
            {
                "parameter": parameter.replace("_", " "),
                "parameter_key": parameter,
                "old": old,
                "previous": previous,
                "current": current,
                "latest_pct": latest_pct,
                "total_pct": total_pct,
                "severity": max(latest_pct, total_pct),
                "reason": TREND_REASON_RULES.get(parameter, {}).get("why", ""),
                "cause": TREND_REASON_RULES.get(parameter, {}).get("cause", ""),
                "action": TREND_REASON_RULES.get(parameter, {}).get("action", ""),
            }
        )
    return sorted(points, key=lambda item: float(item["severity"]), reverse=True)


def dynamic_predict_payload(values: dict[str, float]) -> dict:
    model = load_dynamic_model()
    features = np.array([[float(values[name]) for name in DYNAMIC_INPUT_FEATURES]], dtype=np.float64)
    model_hi = float(predict(model, features)[0][0])
    trend_violations = evaluate_trend_capping(values, TREND_CAPPING_RULES)
    change_points = trend_change_points(values)
    current_values = dynamic_current_values(values)
    current_row = {name: float(current_values.get(name, DEFAULT_INPUT[name])) for name in RAW_FEATURES}
    current_status = gas_conditions(current_row)
    current_violations = operating_limit_violations(current_values)
    reasons = issue_reasons(current_row, current_values, current_status, current_violations)
    existing_reason_keys = {(item["parameter"], item["value"]) for item in reasons}
    for item in hard_cap_reasons(current_violations):
        if (item["parameter"], item["value"]) not in existing_reason_keys:
            reasons.append(item)
    historical_reasons: list[dict[str, str]] = []
    for time_key in ("Tminus2", "Tminus1"):
        historical_values = dynamic_values_for_time(values, time_key)
        historical_row = {name: float(historical_values.get(name, DEFAULT_INPUT[name])) for name in RAW_FEATURES}
        historical_status = gas_conditions(historical_row)
        historical_violations = operating_limit_violations(historical_values)
        historical_items = issue_reasons(historical_row, historical_values, historical_status, historical_violations)
        historical_items.extend(hard_cap_reasons(historical_violations))
        for item in historical_items:
            if str(item.get("condition", "")).lower() not in {"critical", "poor/critical"}:
                continue
            year_title = TIME_POINT_LABELS[time_key]
            historical_reasons.append(
                {
                    "source": "historical",
                    "parameter": f"{year_title} - {item['parameter']}",
                    "value": item["value"],
                    "condition": item["condition"],
                    "reason": (
                        f"{year_title} had a critical threshold violation for {item['parameter']}. "
                        f"{item['reason']} This historical critical value is considered in the trend-based health assessment."
                    ),
                }
            )
    reasons.extend(historical_reasons)
    for item in trend_violations:
        parameter = str(item["parameter"])
        reason_rule = TREND_REASON_RULES.get(parameter, {})
        reason = reason_rule.get("message") or str(item.get("reason", "")) or "The parameter has deteriorated beyond the configured trend limit."
        action = reason_rule.get("action", "")
        previous = float(item["previous"])
        current = float(item["current"])
        old = float(item["old"])
        reasons.append(
            {
                "source": "trend",
                "parameter": parameter.replace("_", " "),
                "value": f"Year 1: {old:g}, Year 2: {previous:g}, Current: {current:g}",
                "condition": "Critical",
                "reason": (
                    f"{parameter} shows a severe deteriorating trend over the last three readings. "
                    f"Latest-year worsening is {float(item['latest_pct']):.1f}% and cumulative worsening is "
                    f"{float(item['total_pct']):.1f}%. {reason} {action} The transformer should be treated as Critical "
                    "until the trend is verified and corrected."
                ),
            }
        )
    hard_cap_active = bool(current_violations) or bool(trend_violations)
    dynamic_hi = 0.0 if hard_cap_active else model_hi
    condition = classify_hi(dynamic_hi)
    return {
        "input_count": len(DYNAMIC_INPUT_FEATURES),
        "dynamic_health_index": round(dynamic_hi, 2),
        "model_dynamic_health_index": round(model_hi, 2),
        "condition": condition,
        "recommendation": recommendation(condition, reasons),
        "current_hard_cap_override": bool(current_violations),
        "trend_hard_cap_override": bool(trend_violations),
        "current_violations": current_violations,
        "trend_violations": trend_violations,
        "trend_change_points": change_points,
        "trend_reason_points": [
            {
                "source": "trend",
                "parameter": str(item["parameter"]).replace("_", " "),
                "value": f"Year 1: {float(item['old']):g}, Year 2: {float(item['previous']):g}, Current: {float(item['current']):g}",
                "condition": "Critical",
                "reason": (
                    f"Latest-year worsening: {float(item['latest_pct']):.1f}%. "
                    f"Total 3-year worsening: {float(item['total_pct']):.1f}%. "
                    f"{(TREND_REASON_RULES.get(str(item['parameter']), {}).get('message') or str(item.get('reason', 'The trend is outside the configured limit.')))} "
                    f"{TREND_REASON_RULES.get(str(item['parameter']), {}).get('action', '')}"
                ),
            }
            for item in trend_violations
        ],
        "historical_reason_points": historical_reasons,
        "reason_points": reasons,
    }


def render_trend_page() -> str:
    sections = dynamic_input_sections_html()
    page = render_index()
    start = page.find("<style>")
    end = page.find("</style>") + len("</style>")
    style = page[start:end]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trend-Based Health Index</title>
  {style}
  <style>
    .trend-loading {{
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      place-items: center;
      padding: 24px;
      background:
        linear-gradient(rgba(3, 7, 12, .88), rgba(3, 7, 12, .94)),
        repeating-linear-gradient(90deg, rgba(47,230,255,.08) 0 1px, transparent 1px 110px),
        repeating-linear-gradient(0deg, rgba(64,255,168,.06) 0 1px, transparent 1px 110px);
      backdrop-filter: blur(10px);
    }}
    .trend-loading.show {{ display: grid; }}
    .grid-loader {{
      width: min(620px, 100%);
      border: 1px solid rgba(47, 230, 255, .34);
      border-radius: 12px;
      background: rgba(7, 18, 28, .92);
      box-shadow: 0 0 60px rgba(47,230,255,.18), inset 0 0 40px rgba(64,255,168,.05);
      padding: 30px;
      text-align: center;
      overflow: hidden;
      position: relative;
    }}
    .grid-loader::before {{
      content: "";
      position: absolute;
      left: -30%;
      right: -30%;
      top: 50%;
      height: 2px;
      background: linear-gradient(90deg, transparent, #2fe6ff, #40ffa8, transparent);
      animation: scanLine 1.4s ease-in-out infinite;
      opacity: .85;
    }}
    .tower-mark {{
      width: 86px;
      height: 86px;
      margin: 0 auto 18px;
      border: 2px solid rgba(47,230,255,.6);
      border-radius: 50%;
      display: grid;
      place-items: center;
      color: #40ffa8;
      font-weight: 900;
      font-size: 28px;
      box-shadow: 0 0 34px rgba(47,230,255,.24);
      animation: gridPulse 1.2s ease-in-out infinite;
    }}
    .grid-loader strong {{
      display: block;
      font-size: clamp(24px, 4vw, 34px);
      margin-bottom: 10px;
      color: #eefaff;
    }}
    .grid-loader span {{
      display: block;
      color: #9fd4df;
      line-height: 1.55;
    }}
    .power-steps {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 22px;
    }}
    .power-steps i {{
      display: block;
      min-height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
      position: relative;
    }}
    .power-steps i::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, #2fe6ff, #40ffa8);
      transform: translateX(-100%);
      animation: chargeStep 1.5s ease-in-out infinite;
    }}
    .power-steps i:nth-child(2)::after {{ animation-delay: .18s; }}
    .power-steps i:nth-child(3)::after {{ animation-delay: .36s; }}
    @keyframes gridPulse {{
      0%, 100% {{ transform: scale(1); box-shadow: 0 0 28px rgba(47,230,255,.18); }}
      50% {{ transform: scale(1.06); box-shadow: 0 0 48px rgba(64,255,168,.3); }}
    }}
    @keyframes scanLine {{
      0%, 100% {{ transform: translateY(-38px); opacity: .35; }}
      50% {{ transform: translateY(42px); opacity: 1; }}
    }}
    @keyframes chargeStep {{
      0% {{ transform: translateX(-100%); }}
      55%, 100% {{ transform: translateX(0); }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><img class="brand-logo" src="{LOGO_URL}" alt="Power Asset Intelligence logo"><span>Power Asset Intelligence</span></div>
      <div class="system-chip">3-Year Dynamic HI Console</div>
    </header>
    <section class="modal open" style="position:static; display:flex; background:transparent; backdrop-filter:none; padding:18px 0;" aria-hidden="false">
      <div class="modal-card" role="dialog" aria-modal="false">
        <div class="modal-head">
          <h2>Trend-Based Health Index Input</h2>
          <div class="modal-actions">
            <button class="home-btn" type="button" onclick="window.location.href='/'">Main Page</button>
            <button class="reset-btn" type="button" id="trendResetBtn">Reset</button>
          </div>
        </div>
        <form id="trendForm" method="post" action="/trend/result">
          <section class="form-section">
            <h3>Transformer Details</h3>
            <div class="fields">
              <label><span>Substation Name</span><input name="SubstationName" type="text" value="Substation-1"></label>
              <label><span>Username</span><input name="Username" type="text" value=""></label>
              <label><span>Designation</span><input name="Designation" type="text" value=""></label>
              <label><span>Organisation</span><input name="Organisation" type="text" value=""></label>
              <label><span>Voltage Ratio (kV)</span><input name="VoltageRatio" type="text" value="220/132 kV"></label>
            </div>
          </section>
          {sections}
          <div class="actions">
            <button class="reset-btn" type="button">Reset</button>
            <button class="predict-btn" type="submit">Predict Trend-Based HI</button>
          </div>
        </form>
        <footer class="modal-footer"><a href="/contact">Created by Satya</a></footer>
      </div>
    </section>
  </main>
  <section class="trend-loading" id="trendLoading" aria-hidden="true">
    <div class="grid-loader">
      <div class="tower-mark">HV</div>
      <strong>Analyzing 3-Year Transformer Trend</strong>
      <span>Comparing DGA, insulation resistance, bushing, winding and thermal signals across the selected years...</span>
      <div class="power-steps"><i></i><i></i><i></i></div>
    </div>
  </section>
  <script>
    const trendForm = document.getElementById('trendForm');
    const trendLoading = document.getElementById('trendLoading');
    function numericValue(name) {{
      const field = trendForm.elements[name];
      const value = Number.parseFloat(field ? field.value : '');
      return Number.isFinite(value) ? value : 0;
    }}
    function calculatePi(tenMinuteName, oneMinuteName) {{
      const tenMinute = numericValue(tenMinuteName);
      const oneMinute = numericValue(oneMinuteName);
      if (tenMinute <= 0 || oneMinute <= 0) return 0;
      return tenMinute / oneMinute;
    }}
    function updateTrendPi(timeKey) {{
      const pairs = [
        ['PI_HV_' + timeKey, 'IR_HV_E_10min_' + timeKey, 'IR_HV_E_1min_' + timeKey],
        ['PI_LV_' + timeKey, 'IR_LV_E_10min_' + timeKey, 'IR_LV_E_1min_' + timeKey],
        ['PI_TV_' + timeKey, 'IR_TV_E_10min_' + timeKey, 'IR_TV_E_1min_' + timeKey]
      ];
      pairs.forEach(([piName, tenName, oneName]) => {{
        if (trendForm.elements[piName]) trendForm.elements[piName].value = calculatePi(tenName, oneName).toFixed(3);
      }});
    }}
    ['Tminus2', 'Tminus1', 'T0_Current'].forEach(timeKey => {{
      ['IR_HV_E_1min_', 'IR_LV_E_1min_', 'IR_TV_E_1min_', 'IR_HV_E_10min_', 'IR_LV_E_10min_', 'IR_TV_E_10min_'].forEach(prefix => {{
        const field = trendForm.elements[prefix + timeKey];
        if (field) field.addEventListener('input', () => updateTrendPi(timeKey));
      }});
      updateTrendPi(timeKey);
    }});
    function clearTrendForm() {{
      Array.from(trendForm.elements).forEach(field => {{
        if (['button', 'submit'].includes(field.type)) return;
        field.value = '';
      }});
    }}
    document.querySelectorAll('.reset-btn').forEach(button => button.addEventListener('click', clearTrendForm));
    trendForm.addEventListener('submit', event => {{
      event.preventDefault();
      trendLoading.classList.add('show');
      trendLoading.setAttribute('aria-hidden', 'false');
      setTimeout(() => trendForm.submit(), 2200);
    }});
  </script>
</body>
</html>"""


def render_contact_page() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Contact Us - Power Asset Intelligence</title>
  <style>
    :root {{
      --bg: #03070c;
      --panel: rgba(9, 22, 34, .86);
      --line: rgba(47, 230, 255, .24);
      --text: #eefaff;
      --muted: #9bb2c0;
      --cyan: #2fe6ff;
      --green: #40ffa8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 18%, rgba(47,230,255,.15), transparent 32%),
        radial-gradient(circle at 85% 8%, rgba(64,255,168,.12), transparent 28%),
        linear-gradient(135deg, #041017, #03070c 62%, #06130e);
      overflow-x: hidden;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        repeating-linear-gradient(90deg, rgba(47,230,255,.07) 0 1px, transparent 1px 108px),
        repeating-linear-gradient(0deg, rgba(64,255,168,.05) 0 1px, transparent 1px 108px);
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.9), transparent 82%);
    }}
    .shell {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 46px; position: relative; }}
    .topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 34px; }}
    .brand {{ display: flex; align-items: center; gap: 14px; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; font-size: 13px; }}
    .brand-logo {{ width: 72px; height: 72px; object-fit: contain; border-radius: 50%; background: rgba(255,255,255,.94); padding: 3px; box-shadow: 0 0 28px rgba(47,230,255,.18); }}
    .home-btn {{ border: 1px solid rgba(47,230,255,.42); border-radius: 8px; padding: 12px 16px; color: var(--text); background: rgba(255,255,255,.07); text-decoration: none; font-weight: 800; }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(145deg, rgba(9,22,34,.92), rgba(5,14,22,.96));
      box-shadow: 0 30px 90px rgba(0,0,0,.48), 0 0 46px rgba(47,230,255,.08);
      overflow: hidden;
    }}
    .hero {{
      padding: clamp(26px, 5vw, 52px);
      border-bottom: 1px solid rgba(47,230,255,.18);
    }}
    .eyebrow {{ color: var(--green); font-weight: 900; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: clamp(36px, 6vw, 68px); line-height: 1; }}
    .subtitle {{ max-width: 760px; color: #d8edf3; font-size: 20px; line-height: 1.6; margin: 22px 0 0; }}
    .content {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 22px; padding: clamp(22px, 4vw, 38px); }}
    .panel {{ border: 1px solid rgba(255,255,255,.1); border-radius: 10px; background: rgba(255,255,255,.04); padding: 22px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 24px; }}
    .panel p {{ color: #dcecf2; line-height: 1.72; margin: 0 0 14px; }}
    strong {{ color: #fff; }}
    .contact-box {{
      display: grid;
      gap: 12px;
      margin-top: 18px;
      padding: 18px;
      border-radius: 10px;
      background: rgba(47,230,255,.07);
      border: 1px solid rgba(47,230,255,.18);
    }}
    .contact-box small {{ color: var(--muted); font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }}
    .contact-box a {{ color: var(--green); font-size: 22px; font-weight: 900; text-decoration: none; overflow-wrap: anywhere; }}
    .signature {{ margin-top: 18px; color: var(--green); font-size: 22px; font-weight: 900; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .chip {{ border: 1px solid rgba(64,255,168,.24); border-radius: 999px; padding: 9px 12px; color: #c9ffed; background: rgba(64,255,168,.08); }}
    @media (max-width: 820px) {{
      .topbar, .content {{ grid-template-columns: 1fr; flex-direction: column; align-items: flex-start; }}
      .content {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><img class="brand-logo" src="{LOGO_URL}" alt="Power Asset Intelligence logo"><span>Power Asset Intelligence</span></div>
      <a class="home-btn" href="/">Main Page</a>
    </header>
    <section class="card">
      <div class="hero">
        <div class="eyebrow">Contact Us</div>
        <h1>Power Asset Intelligence</h1>
        <p class="subtitle">AI-driven transformer health analytics for smarter, more reliable and resilient power infrastructure.</p>
      </div>
      <div class="content">
        <article class="panel">
          <h2>About Me</h2>
          <p>Hi, I'm <strong>Satya Prakash Mishra</strong>, an <strong>Assistant Engineer at UPPTCL</strong>.</p>
          <p>I'm passionate about combining <strong>Electrical Engineering</strong> with <strong>Artificial Intelligence</strong> to build smarter, more reliable power systems. My focus is on predictive maintenance and intelligent asset management.</p>
          <p><strong>Power Asset Intelligence</strong> is my initiative to develop AI-driven solutions that help utilities predict failures before they happen and make power infrastructure more intelligent, efficient, and resilient.</p>
          <div class="signature">Building the future of intelligent power systems.</div>
        </article>
        <aside class="panel">
          <h2>Get In Touch</h2>
          <p>For feedback, collaboration, technical discussion, or suggestions about the Transformer Health Index system, you can contact me directly.</p>
          <div class="contact-box">
            <small>Email</small>
            <a href="mailto:satyapscience@gmail.com">satyapscience@gmail.com</a>
          </div>
          <div class="chips">
            <span class="chip">Predictive Maintenance</span>
            <span class="chip">Transformer HI</span>
            <span class="chip">AI for Power Systems</span>
            <span class="chip">Asset Intelligence</span>
          </div>
        </aside>
      </div>
    </section>
  </main>
</body>
</html>"""


def merge_reason_items(items: list[dict[str, str]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for item in items:
        parameter = str(item.get("parameter", "")).strip()
        if not parameter:
            continue
        entry = merged.setdefault(
            parameter,
            {
                "parameter": parameter,
                "value": item.get("value", ""),
                "condition": item.get("condition", "Critical"),
                "recommendations": [],
            },
        )
        reason = str(item.get("reason", "")).strip()
        if reason and reason not in entry["recommendations"]:
            entry["recommendations"].append(reason)
        if str(item.get("condition", "")).lower() == "critical":
            entry["condition"] = "Critical"
    return list(merged.values())


def recommendation_points_html(recommendations: list[str]) -> str:
    if not recommendations:
        return ""
    items = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in recommendations)
    return f'<ul class="reason-list">{items}</ul>'


def render_trend_result_page(query: dict[str, list[str]]) -> str:
    values = parse_dynamic_values(query)
    data = dynamic_predict_payload(values)
    years = trend_year_labels(query)
    substation_name = query_text(query, "SubstationName", "Substation-1")
    username = query_text(query, "Username", "")
    designation = query_text(query, "Designation", "")
    voltage_ratio = query_text(query, "VoltageRatio", "220/132 kV")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    substation_name = html.escape(query_text(query, "SubstationName", "Substation-1"))
    voltage_ratio = html.escape(query_text(query, "VoltageRatio", "220/132 kV"))
    current_reason_items = merge_reason_items([item for item in data.get("reason_points", []) if item.get("source") != "trend"])
    reason_cards = "\n".join(
        f"""
        <article class="reason-card">
          <div class="reason-top">
            <strong>{html.escape(str(item["parameter"])).replace("Year 1 (Oldest)", years["Tminus2"]).replace("Year 2", years["Tminus1"])}</strong>
            <span class="condition-pill">{html.escape(item["condition"])}</span>
          </div>
          <dl>
            <div><dt>Observed value</dt><dd>{html.escape(item["value"])}</dd></div>
            <div><dt>Recommendation</dt><dd>{recommendation_points_html([str(reason).replace("Year 1 (Oldest)", years["Tminus2"]).replace("Year 2", years["Tminus1"]) for reason in item["recommendations"]])}</dd></div>
          </dl>
        </article>
        """
        for item in current_reason_items
    )
    if not reason_cards:
        reason_cards = '<article class="reason-card ok"><div class="reason-top"><strong>No abnormal parameter found</strong><span class="condition-pill">Normal</span></div><p>Current-year values are within the active hard-capping limits. Continue routine trend monitoring.</p></article>'
    trend_reason_cards = "\n".join(
        f"""
        <article class="reason-card">
          <div class="reason-top">
            <strong>{html.escape(item["parameter"])}</strong>
            <span class="condition-pill">{html.escape(item["condition"])}</span>
          </div>
          <dl>
            <div><dt>Trend values</dt><dd>{html.escape(item["value"]).replace("Year 1", years["Tminus2"]).replace("Year 2", years["Tminus1"]).replace("Current", years["T0_Current"])}</dd></div>
            <div><dt>Assessment</dt><dd>{recommendation_points_html([part.strip() for part in str(item["reason"]).split(". ") if part.strip()])}</dd></div>
          </dl>
        </article>
        """
        for item in data.get("trend_reason_points", [])
    )
    if not trend_reason_cards:
        trend_reason_cards = '<article class="reason-card ok"><div class="reason-top"><strong>No severe trend deterioration found</strong><span class="condition-pill">Clear</span></div><p>The three-year trend values are within the configured trend-based capping policy.</p></article>'
    change_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(str(item["parameter"]))}</td>
          <td>{float(item["old"]):g}</td>
          <td>{float(item["previous"]):g}</td>
          <td>{float(item["current"]):g}</td>
          <td>{float(item["latest_pct"]):.1f}%</td>
          <td>{float(item["total_pct"]):.1f}%</td>
          <td>{html.escape(str(item.get("reason", "")))}</td>
          <td>{html.escape(str(item.get("action", "")))}</td>
        </tr>
        """
        for item in data.get("trend_change_points", [])[:20]
    )
    if not change_rows:
        change_rows = '<tr><td colspan="8">No deteriorating trend detected in active trend parameters.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trend-Based Health Result</title>
  <style>
    body {{ margin:0; min-height:100vh; font-family: Inter, Segoe UI, Arial, sans-serif; background:#03070c; color:#eefaff; padding:24px; }}
    .layout {{ width:min(1120px,100%); margin:0 auto; }}
    .top {{ display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start; margin-bottom:18px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(30px,5vw,58px); }}
    .sub, .muted {{ color:#9bb2c0; }}
    .score {{ text-align:right; }}
    .score small {{ display:block; color:#9bb2c0; }}
    .score strong {{ display:block; font-size:64px; color:#40ffa8; }}
    .top-actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }}
    .result-title-row {{ display:flex; align-items:center; gap:14px; }}
    .result-logo {{ width:76px; height:76px; object-fit:contain; border-radius:50%; background:rgba(255,255,255,.94); padding:3px; }}
    .result-btn {{ color:#021019; background:linear-gradient(135deg,#2fe6ff,#40ffa8); padding:12px 16px; border-radius:8px; text-decoration:none; font-weight:800; }}
    .result-btn.secondary {{ color:#eefaff; background:rgba(255,255,255,.08); border:1px solid rgba(47,230,255,.35); }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; }}
    .panel {{ border:1px solid rgba(47,230,255,.22); border-radius:10px; background:rgba(9,22,34,.86); padding:18px; box-shadow:0 20px 70px rgba(0,0,0,.32); }}
    .panel h2 {{ margin:0 0 14px; }}
    .detail small {{ display:block; color:#9bb2c0; margin-bottom:6px; }}
    .detail strong {{ font-size:22px; }}
    .recommendation-summary {{ margin:0 0 14px; padding:13px 15px; border:1px solid rgba(47,230,255,.18); border-radius:8px; background:rgba(47,230,255,.06); line-height:1.55; }}
    .reason-card {{ border:1px solid rgba(255,93,108,.28); border-radius:8px; padding:16px; margin-bottom:12px; background:rgba(255,93,108,.08); }}
    .reason-card.ok {{ border-color:rgba(64,255,168,.25); background:rgba(64,255,168,.08); }}
    .reason-top {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:12px; }}
    .condition-pill {{ padding:4px 10px; border-radius:999px; background:rgba(255,220,160,.12); color:#ffdca0; font-size:12px; font-weight:800; text-transform:uppercase; white-space:nowrap; }}
    dl {{ display:grid; gap:10px; margin:0; }}
    dl div {{ display:grid; grid-template-columns:140px 1fr; gap:12px; }}
    dt {{ color:#9bb2c0; font-size:12px; font-weight:800; text-transform:uppercase; }}
    dd {{ margin:0; line-height:1.55; }}
    .reason-list {{ margin:0; padding-left:20px; }}
    .reason-list li {{ margin-bottom:8px; line-height:1.55; }}
    .trend-table {{ width:100%; border-collapse:collapse; overflow:hidden; border-radius:8px; }}
    .trend-table th, .trend-table td {{ border-bottom:1px solid rgba(255,255,255,.1); padding:10px 9px; text-align:left; vertical-align:top; }}
    .trend-table th {{ color:#9bb2c0; font-size:12px; text-transform:uppercase; }}
    .trend-table td {{ color:#eefaff; line-height:1.45; }}
    @media (max-width:760px) {{ .top, .grid {{ grid-template-columns:1fr; }} .score {{ text-align:left; }} dl div {{ grid-template-columns:1fr; gap:4px; }} }}
  </style>
</head>
<body>
  <main class="layout">
    <section class="top">
      <div>
        <div class="result-title-row"><img class="result-logo" src="{LOGO_URL}" alt="Power Asset Intelligence logo"><h1>Trend-Based Health Result</h1></div>
        <div class="sub">3-year dynamic Health Index prediction</div>
        <div class="top-actions">
          <a class="result-btn" href="/trend">Back to Trend Input</a>
          <a class="result-btn secondary" href="/">Main Page</a>
          <a class="result-btn secondary" href="/contact">Contact Us</a>
          <form method="post" action="/trend/result.pdf" style="margin:0;">
            {hidden_form_fields(query)}
            <button class="result-btn" type="submit" style="border:0; cursor:pointer;">Download PDF</button>
          </form>
        </div>
      </div>
      <div class="score"><small>Dynamic Health Index</small><strong>{data["dynamic_health_index"]:.2f}</strong></div>
    </section>
    <section class="grid">
      <div class="panel detail"><small>Condition</small><strong>{html.escape(data["condition"])}</strong></div>
      <div class="panel detail"><small>Model Inputs</small><strong>{data["input_count"]} parameters</strong></div>
      <div class="panel detail"><small>Substation</small><strong>{substation_name}</strong></div>
      <div class="panel detail"><small>Voltage Ratio</small><strong>{voltage_ratio}</strong></div>
    </section>
    <section class="panel">
      <h2>Recommendation</h2>
      <p class="recommendation-summary">{html.escape(data["recommendation"])}</p>
    </section>
    <section class="panel" style="margin-top:18px;">
      <h2>Trend-Based Risk Factors</h2>
      <table class="trend-table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>{years["Tminus2"]}</th>
            <th>{years["Tminus1"]}</th>
            <th>{years["T0_Current"]}</th>
            <th>Latest Change</th>
            <th>3-Year Change</th>
            <th>Reason</th>
            <th>Recommended Action</th>
          </tr>
        </thead>
        <tbody>
          {change_rows}
        </tbody>
      </table>
    </section>
    <section class="panel" style="margin-top:18px;">
      <h2>Trend-Based Critical Factors</h2>
      {trend_reason_cards}
    </section>
    <section class="panel" style="margin-top:18px;">
      <h2>Critical Threshold Violations</h2>
      {reason_cards}
    </section>
  </main>
</body>
</html>"""


def render_trend_result_pdf(query: dict[str, list[str]]) -> tuple[bytes, str]:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    values = parse_dynamic_values(query)
    data = dynamic_predict_payload(values)
    years = trend_year_labels(query)
    substation_name = query_text(query, "SubstationName", "Substation-1")
    username = query_text(query, "Username", "")
    designation = query_text(query, "Designation", "")
    organisation = query_text(query, "Organisation", "")
    voltage_ratio = query_text(query, "VoltageRatio", "220/132 kV")
    generated_at = datetime.now()
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    report_id = report_id_from_datetime(generated_at)
    site_url = public_base_url()
    buffer = io.BytesIO()
    filename = f"trend_health_index_{generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=58, bottomMargin=48)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#12313d")
    styles["Heading1"].textColor = colors.HexColor("#0f766e")
    styles["Heading2"].textColor = colors.HexColor("#12313d")
    title_style = ParagraphStyle("TrendCenteredTitle", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica", fontSize=18, leading=24, textColor=colors.HexColor("#12313d"))
    heading_style = ParagraphStyle("TrendCenteredHeading", parent=styles["Heading2"], alignment=TA_CENTER, fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor("#12313d"), spaceAfter=10)

    def draw_header_footer(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFont("Helvetica-Bold", 42)
        canvas.setFillColor(colors.HexColor("#12313d"))
        try:
            canvas.setFillAlpha(0.045)
        except AttributeError:
            pass
        canvas.translate(width / 2, height / 2)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "POWER ASSET INTELLIGENCE")
        canvas.restoreState()
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#12313d"))
        canvas.rect(0, height - 42, width, 42, fill=1, stroke=0)
        if LOGO_PATH.exists():
            canvas.drawImage(str(LOGO_PATH), 36, height - 40, width=34, height=34, preserveAspectRatio=True, mask="auto")
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(78, height - 26, "Transformer Health Monitoring System")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(width - 36, height - 26, "Trend-Based Health Index Report")
        canvas.setStrokeColor(colors.HexColor("#d5e7ed"))
        canvas.line(36, 34, width - 36, 34)
        canvas.setFillColor(colors.HexColor("#0f766e"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(36, 20, site_url)
        canvas.linkURL(site_url, (36, 17, 132, 29), relative=0)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(width / 2, 20, f"Generated: {timestamp}")
        canvas.drawRightString(width - 36, 20, f"Page {document.page}")
        canvas.restoreState()

    card_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fbfc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8dce3")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e5ea")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]
    )
    story = [
        Paragraph("<u>Transformer Health Monitoring System</u>", title_style),
        Paragraph("<u>Trend-Based Health Index Result</u>", heading_style),
        Spacer(1, 12),
        Table(
            [
                ["Dynamic Health Index", f"{data['dynamic_health_index']:.2f}"],
                ["Condition", html.escape(data["condition"])],
                ["Model Inputs", f"{data['input_count']} parameters"],
                ["Substation", html.escape(substation_name)],
                ["Username", html.escape(username) or "-"],
                ["Designation", html.escape(designation) or "-"],
                ["Organisation", html.escape(organisation) or "-"],
                ["Voltage Ratio", html.escape(voltage_ratio)],
                ["Generated", html.escape(timestamp)],
            ],
            colWidths=[220, 260],
            style=card_style,
        ),
        Spacer(1, 12),
        Paragraph("Recommendation", styles["Heading2"]),
        Paragraph(html.escape(data["recommendation"]), styles["BodyText"]),
        PageBreak(),
    ]

    story.append(Paragraph("<u>Trend-Based Risk Factors</u>", heading_style))
    risk_rows = [["Parameter", years["Tminus2"], years["Tminus1"], years["T0_Current"], "Latest %", "3-Year %"]]
    for item in data.get("trend_change_points", [])[:30]:
        risk_rows.append(
            [
                str(item["parameter"]),
                f"{float(item['old']):g}",
                f"{float(item['previous']):g}",
                f"{float(item['current']):g}",
                f"{float(item['latest_pct']):.1f}%",
                f"{float(item['total_pct']):.1f}%",
            ]
        )
    risk_table = Table(risk_rows, repeatRows=1)
    risk_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9edf7")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8cbd1")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 5), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([risk_table, PageBreak()])

    story.append(Paragraph("<u>Trend-Based Critical Factors</u>", heading_style))
    for item in data.get("trend_reason_points", []):
        story.append(Paragraph(f"<b>{html.escape(item['parameter'])}</b>", styles["Heading3"]))
        story.append(Paragraph(f"Values: {html.escape(item['value']).replace('Year 1', years['Tminus2']).replace('Year 2', years['Tminus1']).replace('Current', years['T0_Current'])}", styles["BodyText"]))
        for part in [part.strip() for part in str(item["reason"]).split(". ") if part.strip()]:
            story.append(Paragraph(f"- {html.escape(part)}", styles["BodyText"]))
        story.append(Spacer(1, 8))
    if not data.get("trend_reason_points"):
        story.append(Paragraph("No severe trend deterioration found.", styles["BodyText"]))
    story.append(PageBreak())

    story.append(Paragraph("<u>Critical Threshold Violations</u>", heading_style))
    current_items = merge_reason_items([item for item in data.get("reason_points", []) if item.get("source") != "trend"])
    if current_items:
        for item in current_items:
            parameter_text = html.escape(str(item["parameter"])).replace("Year 1 (Oldest)", years["Tminus2"]).replace("Year 2", years["Tminus1"])
            story.append(Paragraph(f"<b>{parameter_text}: {html.escape(str(item['value']))}</b>", styles["Heading3"]))
            for reason in item["recommendations"]:
                reason_text = html.escape(str(reason)).replace("Year 1 (Oldest)", years["Tminus2"]).replace("Year 2", years["Tminus1"])
                story.append(Paragraph(f"- {reason_text}", styles["BodyText"]))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No critical threshold violation found.", styles["BodyText"]))
    story.append(PageBreak())

    trend_critical_parameters = {str(item.get("parameter", "")) for item in data.get("trend_violations", [])}
    trend_warning_parameters = {str(item.get("parameter_key", item.get("parameter", ""))) for item in data.get("trend_change_points", [])}
    current_critical_labels = {str(item.get("parameter", "")).lower() for item in current_items}
    for time_key, title in TIME_POINT_LABELS.items():
        story.append(Paragraph(f"<u>Input Parameters - {years[time_key]}</u>", heading_style))
        rows = [["Parameter", "Value"]]
        highlight_rows: dict[int, str] = {}
        for key in [name for name in DYNAMIC_INPUT_FEATURES if name.endswith(f"_{time_key}")]:
            label, unit = dynamic_input_label(key)
            base, _ = split_dynamic_feature(key)
            row_index = len(rows)
            rows.append([f"{label} ({unit})" if unit else label, f"{values[key]:g}"])
            label_key = label.lower()
            if base in trend_critical_parameters or (time_key == "T0_Current" and any(label_key.startswith(item) or item in label_key for item in current_critical_labels)):
                highlight_rows[row_index] = "critical"
            elif base in trend_warning_parameters:
                highlight_rows[row_index] = "warning"
        table = Table(rows, colWidths=[260, 220], repeatRows=1)
        input_commands = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5e9")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8cbd1")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 5), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        for row_index, severity in highlight_rows.items():
            input_commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f8d7da" if severity == "critical" else "#fff3cd")))
        table.setStyle(TableStyle(input_commands))
        story.append(table)
        if time_key != "T0_Current":
            story.append(PageBreak())

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    pdf_body = apply_cover_template(
        buffer.getvalue(),
        report_type="Trend",
        username=username,
        organisation=organisation,
        substation_name=substation_name,
        voltage_ratio=voltage_ratio,
        timestamp=timestamp,
        report_id=report_id,
    )
    return pdf_body, filename


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = render_index().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/static/"):
            file_path = (STATIC_DIR / parsed.path.removeprefix("/static/")).resolve()
            if STATIC_DIR.resolve() in file_path.parents and file_path.exists() and file_path.is_file():
                body = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            json_response(self, {"error": "Static file not found"}, status=404)
            return
        if parsed.path == "/result":
            query = parse_qs(parsed.query)
            body = render_result_page(query, parsed.query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/trend":
            body = render_trend_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/contact":
            body = render_contact_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/trend/result":
            query = parse_qs(parsed.query)
            body = render_trend_result_page(query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/trend/result.pdf":
            query = parse_qs(parsed.query)
            body, filename = render_trend_result_pdf(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/result.pdf":
            query = parse_qs(parsed.query)
            body, filename = render_result_pdf(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/predict":
            query = parse_qs(parsed.query)
            values = {key: float(query.get(key, [DEFAULT_INPUT[key]])[0]) for key in DEFAULT_INPUT}
            json_response(self, predict_payload(values))
            return
        if parsed.path == "/api/trend-predict":
            query = parse_qs(parsed.query)
            values = parse_dynamic_values(query)
            json_response(self, dynamic_predict_payload(values))
            return
        json_response(self, {"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/result":
            query = parse_form_body(self)
            body = render_result_page(query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/result.pdf":
            query = parse_form_body(self)
            body, filename = render_result_pdf(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/trend/result":
            query = parse_form_body(self)
            body = render_trend_result_page(query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/trend/result.pdf":
            query = parse_form_body(self)
            body, filename = render_trend_result_pdf(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        json_response(self, {"error": "Not found"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Transformer Health Monitoring System running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
