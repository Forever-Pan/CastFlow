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
        result = {
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
        return {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in result.items()
        }


from evaluation.parsing import (
    aligned_arrays,
    cleanup_jsonish,
    extract_answer_text,
    is_number,
    normalize_mapping,
    parse_forecast_mapping,
    parse_forecast_sequence,
    parse_forecast_value,
)


def compute_metrics(
    prediction: dict[str, float], ground_truth: dict[str, float]
) -> dict[str, Any]:
    empty = {
        name: float("nan") for name in ("mse", "mae", "rmse", "nmse", "nmae", "nrmse")
    }
    try:
        pred, truth = aligned_arrays(prediction, ground_truth)
    except (ValueError, TypeError, OverflowError) as exc:
        return {**empty, "n": 0, "valid": False, "error": str(exc)}
    with np.errstate(over="ignore", invalid="ignore"):
        squared = float(np.mean((pred - truth) ** 2))
        absolute = float(np.mean(np.abs(pred - truth)))
        var = float(np.var(truth))
        mean = float(np.mean(truth))
    if not np.isfinite(squared) or not np.isfinite(absolute):
        return {
            **empty,
            "n": 0,
            "valid": False,
            "error": "forecast error exceeds finite numerical range",
        }
    root = float(np.sqrt(squared))
    return {
        "mse": squared,
        "mae": absolute,
        "rmse": root,
        "nmse": squared / var if np.isfinite(var) and var > 0 else float("nan"),
        "nmae": absolute / abs(mean) if abs(mean) > 1e-10 else float("nan"),
        "nrmse": root / abs(mean) if abs(mean) > 1e-10 else float("nan"),
        "n": int(pred.size),
        "valid": True,
        "error": "",
    }


def evaluate_csv(
    csv_path: str | Path,
    answer_col: str = "answer",
    ground_truth_col: str = "ground_truth",
    output_path: str | Path | None = None,
) -> EvaluationSummary:
    from utils.io import atomic_write_csv

    source = Path(csv_path)
    if output_path is not None and source.resolve() == Path(output_path).resolve():
        raise ValueError("evaluation output must differ from its input CSV")
    frame = pd.read_csv(source, keep_default_na=False)
    for name in (answer_col, ground_truth_col):
        if name not in frame.columns:
            raise ValueError(
                f"column {name!r} not found; columns={list(frame.columns)}"
            )
    row_metrics: list[dict[str, Any]] = []
    all_predictions, all_targets = [], []
    for _, row in frame.iterrows():
        pred = parse_forecast_value(row[answer_col])
        truth = parse_forecast_value(row[ground_truth_col])
        metrics = compute_metrics(pred, truth)
        unsuccessful = "success" in frame and str(
            row["success"]
        ).strip().casefold() in {"false", "0", "no"}
        missing = "status" in frame and str(row["status"]).strip().casefold() in {
            "missing",
            "failed",
            "error",
        }
        if unsuccessful or missing:
            metrics = {
                **{
                    name: float("nan")
                    for name in ("mse", "mae", "rmse", "nmse", "nmae", "nrmse")
                },
                "n": 0,
                "valid": False,
                "error": "forecast row is explicitly marked unsuccessful",
            }
        row_metrics.append(metrics)
        if metrics["valid"]:
            p, t = aligned_arrays(pred, truth)
            all_predictions.extend(p.tolist())
            all_targets.extend(t.tolist())
    aggregate = compute_metrics(
        parse_forecast_value(all_predictions), parse_forecast_value(all_targets)
    )
    valid = sum(item["valid"] for item in row_metrics)
    summary = EvaluationSummary(
        total_rows=len(frame),
        valid_rows=valid,
        invalid_rows=len(frame) - valid,
        **{
            name: aggregate[name]
            for name in ("mse", "mae", "rmse", "nmse", "nmae", "nrmse")
        },
    )
    if output_path:
        out = frame.copy()
        for name in (
            "mse",
            "mae",
            "rmse",
            "nmse",
            "nmae",
            "nrmse",
            "n",
            "valid",
            "error",
        ):
            out[name] = [item[name] for item in row_metrics]
        atomic_write_csv(output_path, out)
    return summary


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if np.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")
