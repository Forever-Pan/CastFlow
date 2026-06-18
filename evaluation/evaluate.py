from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EvaluationSummary:
    total_rows: int
    valid_rows: int
    invalid_rows: int
    mse: float
    mae: float
    rmse: float
    nmse: float
    nmae: float
    nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "nmse": self.nmse,
            "nmae": self.nmae,
            "nrmse": self.nrmse,
        }


def extract_answer_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", str(text), flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return str(text).strip()


def parse_forecast_value(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return normalize_mapping(value)
    if isinstance(value, list):
        return {str(i): float(v) for i, v in enumerate(value) if is_number(v)}

    text = extract_answer_text(str(value))
    if not text:
        return {}
    cleaned = cleanup_jsonish(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return normalize_mapping(parsed)
        if isinstance(parsed, list):
            return {str(i): float(v) for i, v in enumerate(parsed) if is_number(v)}
    except json.JSONDecodeError:
        pass

    pairs = re.findall(
        r'"?(\d{4}-\d{2}-\d{2}(?:\s+|T)\d{2}:\d{2}:\d{2}|[A-Za-z_]*\d+)"?\s*:\s*([+-]?\d+(?:\.\d+)?)',
        text,
    )
    if pairs:
        return {key: float(val) for key, val in pairs}

    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return {str(i): float(val) for i, val in enumerate(numbers)}


def cleanup_jsonish(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r",\s*\.\.\.\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*\.\.\.\s*\]", "]", cleaned)
    cleaned = re.sub(r"\[\s*\.\.\.\s*,", "[", cleaned)
    cleaned = cleaned.replace("'", '"')
    return cleaned


def normalize_mapping(value: dict[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in value.items():
        if is_number(val):
            out[str(key)] = float(val)
    return out


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def aligned_arrays(prediction: dict[str, float], ground_truth: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(prediction) & set(ground_truth))
    pred_values: list[float] = []
    truth_values: list[float] = []
    if common:
        for key in common:
            pred_values.append(float(prediction[key]))
            truth_values.append(float(ground_truth[key]))
    else:
        pred_keys = sorted(prediction)
        truth_keys = sorted(ground_truth)
        for p_key, t_key in zip(pred_keys, truth_keys):
            pred_values.append(float(prediction[p_key]))
            truth_values.append(float(ground_truth[t_key]))
    return np.asarray(pred_values, dtype=float), np.asarray(truth_values, dtype=float)


def compute_metrics(prediction: dict[str, float], ground_truth: dict[str, float]) -> dict[str, float]:
    pred, truth = aligned_arrays(prediction, ground_truth)
    if pred.size == 0 or truth.size == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "nmse": float("nan"), "nmae": float("nan"), "nrmse": float("nan"), "n": 0}
    mse = float(np.mean((pred - truth) ** 2))
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(mse))
    var = float(np.var(truth))
    mean = float(np.mean(truth))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nmse": mse / var if var > 0 else float("nan"),
        "nmae": mae / abs(mean) if abs(mean) > 1e-10 else float("nan"),
        "nrmse": rmse / abs(mean) if abs(mean) > 1e-10 else float("nan"),
        "n": int(pred.size),
    }


def evaluate_csv(
    csv_path: str | Path,
    answer_col: str = "answer",
    ground_truth_col: str = "ground_truth",
    output_path: str | Path | None = None,
) -> EvaluationSummary:
    frame = pd.read_csv(csv_path, keep_default_na=False)
    if answer_col not in frame.columns:
        raise ValueError(f"answer column {answer_col!r} not found; columns={list(frame.columns)}")
    if ground_truth_col not in frame.columns:
        raise ValueError(f"ground truth column {ground_truth_col!r} not found; columns={list(frame.columns)}")

    row_metrics: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        pred = parse_forecast_value(row[answer_col])
        truth = parse_forecast_value(row[ground_truth_col])
        metrics = compute_metrics(pred, truth)
        row_metrics.append(metrics)

    valid = [item for item in row_metrics if not np.isnan(item["mse"])]
    summary = EvaluationSummary(
        total_rows=len(frame),
        valid_rows=len(valid),
        invalid_rows=len(frame) - len(valid),
        mse=mean_metric(valid, "mse"),
        mae=mean_metric(valid, "mae"),
        rmse=mean_metric(valid, "rmse"),
        nmse=mean_metric(valid, "nmse"),
        nmae=mean_metric(valid, "nmae"),
        nrmse=mean_metric(valid, "nrmse"),
    )
    if output_path:
        out = frame.copy()
        for key in ("mse", "mae", "rmse", "nmse", "nmae", "nrmse", "n"):
            out[key] = [item[key] for item in row_metrics]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return summary


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if not np.isnan(row[key])]
    return float(np.mean(values)) if values else float("nan")
