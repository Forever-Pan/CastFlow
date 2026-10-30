from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.evaluate import (
    aligned_arrays,
    extract_answer_text,
    parse_forecast_value,
)
from utils.registry import normalize_dataset_name


DATASET_MSE_UPPER_BOUNDS = {
    "BE": 800.0,
    "DE": 350.0,
    "FR": 1000.0,
    "NP": 40.0,
    "PJM": 45.0,
    "ETTh": 12.0,
    "ETTm": 3.5,
    "WP": 3000.0,
    "SP": 50.0,
    "MOPEX": 7.0,
}
DATASET_RELATIVE_UPPER_BOUNDS = {
    "BE": 10.0,
    "DE": 10.0,
    "FR": 10.0,
    "NP": 3.0,
    "PJM": 4.0,
    "ETTh": 2.0,
    "ETTm": 1.0,
    "WP": 30.0,
    "SP": 3.0,
    "MOPEX": 1.5,
}
_DATASET_ALIASES = {name.lower(): name for name in DATASET_MSE_UPPER_BOUNDS}
_DATASET_ALIASES.update(
    {
        "etth1": "ETTh",
        "ettm1": "ETTm",
        "windy": "WP",
        "wind": "WP",
        "sunny": "SP",
        "solar": "SP",
    }
)


def canonical_dataset_name(name: str) -> str:
    """Resolve display and CLI dataset identifiers to the published reward scale."""
    raw = str(name).strip()
    if raw.lower().startswith(("epf_", "ett_")):
        raw = raw.split("_", 1)[1]
    key = normalize_dataset_name(raw).lower()
    if key not in _DATASET_ALIASES:
        raise ValueError(
            f"unknown reward dataset {name!r}; specify one of {', '.join(DATASET_MSE_UPPER_BOUNDS)}"
        )
    return _DATASET_ALIASES[key]


@dataclass(slots=True)
class RewardConfig:
    """Absolute MSE plus clipped improvement over the matched ensemble prior."""

    dataset_name: str = ""
    use_contrastive: bool = True
    relative_scale_factor: float = 5.0
    relative_clip: float = 0.5

    def validate(self) -> None:
        for name, value in (
            ("relative_scale_factor", self.relative_scale_factor),
            ("relative_clip", self.relative_clip),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def compute_reward(
    answer: Any,
    ground_truth: Any,
    dataset_name: str = "",
    tool: Any = None,
    baseline: Any = None,
    config: RewardConfig | None = None,
    **_: Any,
) -> float:
    cfg = config or RewardConfig(dataset_name=dataset_name)
    name = canonical_dataset_name(dataset_name or cfg.dataset_name)
    cfg.validate()
    if cfg.use_contrastive:
        return compute_contrastive_reward(
            serialize_forecast(answer),
            serialize_forecast(ground_truth),
            name,
            tool=tool,
            baseline=baseline,
            config=cfg,
        )
    return compute_absolute_reward(
        serialize_forecast(answer), serialize_forecast(ground_truth), name
    )


def compute_absolute_reward(
    answer: str, ground_truth: str, dataset_name: str = ""
) -> float:
    name = canonical_dataset_name(dataset_name)
    pred, truth, failure = parse_and_validate_answer(answer, ground_truth)
    return (
        failure
        if failure is not None
        else mse_to_reward(calculate_mse(pred, truth), name)
    )


def compute_contrastive_reward(
    answer: str,
    ground_truth: str,
    dataset_name: str = "",
    tool: Any = None,
    baseline: Any = None,
    config: RewardConfig | None = None,
) -> float:
    cfg = config or RewardConfig(dataset_name=dataset_name)
    name = canonical_dataset_name(dataset_name or cfg.dataset_name)
    cfg.validate()
    pred, truth, failure = parse_and_validate_answer(answer, ground_truth)
    if failure is not None:
        return failure
    candidate_mse = calculate_mse(pred, truth)
    absolute = mse_to_reward(candidate_mse, name)
    prior = (
        baseline_prediction_dict(baseline, truth)
        if baseline is not None
        else build_small_model_prediction_dict(tool, truth)
    )
    if not prior:
        return absolute
    prior_mse = calculate_mse(prior, truth)
    if not np.isfinite(prior_mse):
        return absolute
    relative = (
        cfg.relative_scale_factor
        * (prior_mse - candidate_mse)
        / DATASET_RELATIVE_UPPER_BOUNDS[name]
    )
    return float(absolute + np.clip(relative, -cfg.relative_clip, cfg.relative_clip))


def parse_and_validate_answer(
    answer: str, ground_truth: str
) -> tuple[dict[str, float], dict[str, float], float | None]:
    content = extract_answer_from_tags(answer)
    pred, truth = parse_json_column(content), parse_json_column(ground_truth)
    if not truth:
        return pred, truth, -1.0
    if not pred and content.strip() not in {"", "[]", "{}"}:
        return pred, truth, -1.0
    n, horizon = len(pred), len(truth)
    if n < horizon:
        return pred, truth, -0.8
    if n > horizon:
        return pred, truth, max(-0.5 - 0.6 * (n - horizon) / horizon, -0.8)
    if not np.isfinite(calculate_mse(pred, truth)):
        return pred, truth, -1.0
    return pred, truth, None


def mse_to_reward(mse: float, dataset_name: str = "") -> float:
    bound = DATASET_MSE_UPPER_BOUNDS[canonical_dataset_name(dataset_name)]
    if not np.isfinite(mse) or mse < 0:
        raise ValueError("MSE must be finite and nonnegative")
    if mse <= bound:
        return float(1.0 - 0.9 * np.sin(np.pi * mse / (2.0 * bound)))
    return float(0.3 * np.exp(-np.log(3.0) * mse / bound))


def parse_json_column(value: Any) -> dict[str, float]:
    return parse_forecast_value(value)


def list_to_forecast_dict(parsed: list[Any]) -> dict[str, float]:
    return parse_forecast_value(parsed)


def mapping_to_float_dict(parsed: dict[Any, Any]) -> dict[str, float]:
    return parse_forecast_value(parsed)


def regex_forecast_dict(value: str) -> dict[str, float]:
    """Compatibility entry point; incomplete numeric fragments are not forecasts."""
    return parse_forecast_value(value)


def calculate_mse(
    prediction: dict[str, float], ground_truth: dict[str, float]
) -> float:
    if not prediction or not ground_truth or len(prediction) != len(ground_truth):
        return float("nan")
    try:
        pred, truth = aligned_arrays(prediction, ground_truth)
        if pred.size != len(ground_truth) or truth.shape != pred.shape:
            return float("nan")
        if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
            return float("nan")
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.mean(np.square(pred - truth)))
        return value if np.isfinite(value) else float("nan")
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def build_small_model_prediction_dict(
    tool: Any, gt_dict: dict[str, float]
) -> dict[str, float] | None:
    if not gt_dict or tool is None:
        return None
    if isinstance(tool, str):
        try:
            tool = json.loads(tool)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(tool, dict):
        if "model_auxiliary_tool" in tool and isinstance(
            tool["model_auxiliary_tool"], dict
        ):
            return _prior_from_output(tool["model_auxiliary_tool"], gt_dict)
        for field in ("observations", "tools", "tool_calls"):
            if field in tool:
                return build_small_model_prediction_dict(tool[field], gt_dict)
        tool = [tool]
    if not isinstance(tool, list):
        return None
    for item in tool:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name", item.get("name")) == "model_auxiliary_tool":
            output = item.get("output", item)
            return (
                _prior_from_output(output, gt_dict)
                if isinstance(output, dict)
                else None
            )
    return None


def _prior_from_output(
    output: dict[str, Any], truth: dict[str, float]
) -> dict[str, float] | None:
    for field in ("reference_prediction", "ensemble_forecast_baseline"):
        if output.get(field) is not None:
            return baseline_prediction_dict(output[field], truth)
    return None


def baseline_prediction_dict(
    values: Any, gt_dict: dict[str, float]
) -> dict[str, float] | None:
    parsed = parse_forecast_value(
        values.tolist() if isinstance(values, np.ndarray) else values
    )
    if not parsed or len(parsed) != len(gt_dict):
        return None
    try:
        pred, truth = aligned_arrays(parsed, gt_dict)
    except (TypeError, ValueError):
        return None
    if (
        pred.size != len(gt_dict)
        or not np.isfinite(pred).all()
        or not np.isfinite(truth).all()
    ):
        return None
    # Keep the validated key space; alignment may reorder timestamps numerically.
    return parsed


def extract_answer_from_tags(answer_text: str) -> str:
    return extract_answer_text(answer_text)


def serialize_forecast(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return "invalid forecast"
