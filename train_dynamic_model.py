"""Train the 3-year trend-based Transformer Dynamic Health Index model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import APP_DIR, MODEL_DIR, OUTPUT_DIR, metrics, predict, split_data, train_network
from trend_capping import evaluate_trend_capping, load_trend_capping_rules


DEFAULT_DYNAMIC_DATASET = APP_DIR / "data" / "Transformer_Dynamic_HI_3Year_48_Parameter_Dataset.xlsx"
DEFAULT_TREND_CAPPING_DATASET = APP_DIR / "data" / "Transformer_HI_48_Trend_Based_Capping.xlsx"
DYNAMIC_AUGMENTED_DATASET = OUTPUT_DIR / "Dynamic_HI_Training_Set.csv"
DYNAMIC_TARGET = "Dynamic_HI_Final"


def read_dynamic_sheet(dataset_path: Path) -> pd.DataFrame:
    return pd.read_excel(dataset_path, sheet_name="Dynamic_Training_Data")


def dynamic_feature_names(dataset_path: Path) -> list[str]:
    feature_map = pd.read_excel(dataset_path, sheet_name="Dynamic_Feature_Map")
    feature_map = feature_map[feature_map["Use_for_Training"].astype(str).str.lower().eq("yes")]
    return [str(value) for value in feature_map["Model_Column"].tolist()]


def target_as_points(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if float(numeric.max()) <= 1.5:
        return numeric * 100.0
    return numeric


def load_dynamic_training_data(dataset_path: Path, capping_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], int]:
    features = dynamic_feature_names(dataset_path)
    df = read_dynamic_sheet(dataset_path)
    missing = [name for name in (*features, DYNAMIC_TARGET) if name not in df.columns]
    if missing:
        raise ValueError(f"Dynamic dataset is missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=[DYNAMIC_TARGET]).copy()
    for column in features:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=features)
    df["DHI_training_target"] = target_as_points(df[DYNAMIC_TARGET]).clip(0.0, 100.0)
    trend_cap_count = 0
    rules = load_trend_capping_rules(capping_path) if capping_path else []
    if rules:
        capped_targets = []
        cap_flags = []
        cap_reasons = []
        for record in df.to_dict("records"):
            violations = evaluate_trend_capping(record, rules)
            cap_flags.append(1 if violations else 0)
            cap_reasons.append("; ".join(str(item["parameter"]) for item in violations))
            capped_targets.append(0.0 if violations else float(record["DHI_training_target"]))
        df["Trend_Cap_Flag"] = cap_flags
        df["Trend_Cap_Reason"] = cap_reasons
        df["DHI_training_target"] = capped_targets
        trend_cap_count = int(sum(cap_flags))
    x = df.loc[:, features].to_numpy(dtype=np.float64)
    y = df["DHI_training_target"].to_numpy(dtype=np.float64).reshape(-1, 1)
    return x, y, df, features, trend_cap_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DYNAMIC_DATASET)
    parser.add_argument("--capping-dataset", type=Path, default=DEFAULT_TREND_CAPPING_DATASET)
    args = parser.parse_args()

    x, y, df, features, trend_cap_count = load_dynamic_training_data(args.dataset, args.capping_dataset)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = train_network(x_train, y_train)
    model["lookup_x"] = x
    model["lookup_y"] = y
    train_pred = predict(model, x_train)
    test_pred = predict(model, x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_DIR / "dynamic_health_index_model.npz", **model)
    df.to_csv(DYNAMIC_AUGMENTED_DATASET, index=False)

    metadata = {
        "dataset": str(args.dataset.relative_to(APP_DIR)) if args.dataset.is_relative_to(APP_DIR) else str(args.dataset),
        "trend_capping_dataset": str(args.capping_dataset.relative_to(APP_DIR)) if args.capping_dataset.is_relative_to(APP_DIR) else str(args.capping_dataset),
        "augmented_dataset": str(DYNAMIC_AUGMENTED_DATASET.relative_to(APP_DIR)),
        "rows_used": int(len(df)),
        "trend_capped_rows": trend_cap_count,
        "features": features,
        "target": "DHI_training_target",
        "target_rule": "DHI_training_target = 0 when trend-based hard capping rules trigger; otherwise Dynamic_HI_Final.",
        "model_input_count": len(features),
        "train_metrics": metrics(y_train, train_pred),
        "test_metrics": metrics(y_test, test_pred),
        "sample_prediction": float(test_pred[0][0]),
        "sample_actual": float(y_test[0][0]),
    }
    (MODEL_DIR / "dynamic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_DIR / "dynamic_features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
