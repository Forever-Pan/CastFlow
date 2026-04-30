from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


DATASET_MSE_UPPER_BOUNDS = {
    "NP": 40.0,
    "BE": 800.0,
    "DE": 350.0,
    "FR": 1000.0,
    "PJM": 45.0,
    "MOPEX": 7.0,
    "sunny": 50.0,
    "SP": 50.0,
    "windy": 3000.0,
    "WP": 3000.0,
    "ETTh1": 12.0,
    "ETTm1": 3.5,
}


DATASET_RELATIVE_UPPER_BOUNDS = {
    "NP": 3.0,
    "BE": 10.0,
    "DE": 10.0,
    "FR": 10.0,
    "PJM": 4.0,
    "MOPEX": 1.5,
    "sunny": 3.0,
    "SP": 3.0,
    "windy": 30.0,
    "WP": 30.0,
    "ETTh1": 2.0,
    "ETTm1": 1.0,
}


@dataclass(slots=True)
class RewardConfig:
    """Reward settings matching the migrated rl_agent.py semantics."""

    dataset_name: str = ""
    use_contrastive: bool = True
    relative_scale_factor: float = 5.0
    relative_clip: float = 0.5


def compute_reward(
    answer: str | list[float] | np.ndarray | None,
    ground_truth: str | list[float] | np.ndarray,
    dataset_name: str = "",
    tool: str | list[dict[str, Any]] | None = None,
    baseline: list[float] | np.ndarray | None = None,
    config: RewardConfig | None = None,
    **_: Any,
) -> float:
    """Compute the CastFlow RLVR reward.

    This is the CastFlow-native migration of the original RL reward logic:
    invalid JSON is penalized, length mismatch is penalized, valid answers
    receive the dataset-specific MSE reward, and contrastive mode adds a
    clipped relative gain against `reference_prediction`.
    """
    cfg = config or RewardConfig(dataset_name=dataset_name)
    name = dataset_name or cfg.dataset_name
    answer_text = serialize_forecast(answer)
    truth_text = serialize_forecast(ground_truth)
    if cfg.use_contrastive or tool is not None or baseline is not None:
        return compute_contrastive_reward(answer_text, truth_text, name, tool=tool, baseline=baseline, config=cfg)
    return compute_absolute_reward(answer_text, truth_text, name)


def compute_absolute_reward(answer: str, ground_truth: str, dataset_name: str = "") -> float:
    answer_dict, gt_dict, failure = parse_and_validate_answer(answer, ground_truth)
    if failure is not None:
        return failure

    mse = calculate_mse(answer_dict, gt_dict)
    if np.isnan(mse):
        return -1.0
    return mse_to_reward(mse, dataset_name)


def compute_contrastive_reward(
    answer: str,
    ground_truth: str,
    dataset_name: str = "",
    tool: str | list[dict[str, Any]] | None = None,
    baseline: list[float] | np.ndarray | None = None,
    config: RewardConfig | None = None,
) -> float:
    cfg = config or RewardConfig(dataset_name=dataset_name)
    answer_dict, gt_dict, failure = parse_and_validate_answer(answer, ground_truth)
    if failure is not None:
        return failure

    mse_answer = calculate_mse(answer_dict, gt_dict)
    if np.isnan(mse_answer):
        return -1.0

    absolute_reward = mse_to_reward(mse_answer, dataset_name)
    small_model_dict = build_small_model_prediction_dict(tool, gt_dict)
    if small_model_dict is None and baseline is not None:
        small_model_dict = baseline_prediction_dict(baseline, gt_dict)
    if not small_model_dict:
        return absolute_reward

    mse_small = calculate_mse(small_model_dict, gt_dict)
    if np.isnan(mse_small):
        return absolute_reward

    relative_bound = DATASET_RELATIVE_UPPER_BOUNDS.get(dataset_name, 10.0)
    diff = mse_small - mse_answer
    relative_raw = (diff / max(relative_bound, 1e-8)) * cfg.relative_scale_factor
    relative_reward = float(np.clip(relative_raw, -cfg.relative_clip, cfg.relative_clip))
    return float(absolute_reward + relative_reward)


def parse_and_validate_answer(answer: str, ground_truth: str) -> tuple[dict[str, float], dict[str, float], float | None]:
    answer_content = extract_answer_from_tags(answer)
    answer_dict = parse_json_column(answer_content)
    gt_dict = parse_json_column(ground_truth)

    if answer_content and answer_content.strip() and not answer_dict:
        return answer_dict, gt_dict, -1.0
    if not gt_dict:
        return answer_dict, gt_dict, -1.0

    answer_len = len(answer_dict)
    gt_len = len(gt_dict)
    if answer_len > gt_len:
        extra = answer_len - gt_len
        penalty = -0.5 - 0.3 * extra / (0.5 * gt_len)
        return answer_dict, gt_dict, max(penalty, -0.8)
    if answer_len < gt_len:
        return answer_dict, gt_dict, -0.8
    return answer_dict, gt_dict, None


def mse_to_reward(mse: float, dataset_name: str = "") -> float:
    upper_bound = DATASET_MSE_UPPER_BOUNDS.get(dataset_name, 10.0)
    if mse <= upper_bound:
        return float(1.0 - 0.9 * np.sin(np.pi * mse / (2.0 * upper_bound)))
    return float(0.3 * np.exp((-np.log(3.0) * mse) / upper_bound))


def parse_json_column(value: str) -> dict[str, float]:
    if not value or not value.strip():
        return {}

    cleaned = value.strip()
    cleaned = re.sub(r",\s*\.\.\.\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*\.\.\.\s*\]", "]", cleaned)
    cleaned = re.sub(r"\[\s*\.\.\.\s*,", "[", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return list_to_forecast_dict(parsed)
        if isinstance(parsed, dict):
            return mapping_to_float_dict(parsed)
        return {}
    except json.JSONDecodeError:
        return regex_forecast_dict(value)
    except (TypeError, ValueError):
        return {}


def list_to_forecast_dict(parsed: list[Any]) -> dict[str, float]:
    if not parsed:
        return {}
    if all(isinstance(item, (int, float)) for item in parsed):
        return {str(idx): float(value) for idx, value in enumerate(parsed)}
    result: dict[str, float] = {}
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        if len(item) == 1:
            key = next(iter(item))
            value = item[key]
        else:
            key = item.get("timestamp") or item.get("time") or item.get("t") or item.get("date") or str(idx)
            value = item.get("value", item.get("val", item.get("v")))
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def mapping_to_float_dict(parsed: dict[Any, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def regex_forecast_dict(value: str) -> dict[str, float]:
    patterns = [
        r'"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"\s*:\s*([+-]?\d+\.?\d*)',
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):\s*([+-]?\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, value)
        if matches:
            return {timestamp: float(number) for timestamp, number in matches}
    return {}


def calculate_mse(prediction: dict[str, float], ground_truth: dict[str, float]) -> float:
    if not prediction or not ground_truth:
        return float("nan")
    common = sorted(set(prediction) & set(ground_truth))
    if not common:
        return float("nan")
    pred = np.asarray([float(prediction[key]) for key in common], dtype=float)
    truth = np.asarray([float(ground_truth[key]) for key in common], dtype=float)
    if pred.size == 0:
        return float("nan")
    return float(np.mean((pred - truth) ** 2))


def build_small_model_prediction_dict(
    tool: str | list[dict[str, Any]] | None,
    gt_dict: dict[str, float],
) -> dict[str, float] | None:
    if not tool or not gt_dict:
        return None
    tool_data: Any
    if isinstance(tool, str):
        try:
            tool_data = json.loads(tool)
        except Exception:
            return None
    else:
        tool_data = tool
    if isinstance(tool_data, dict):
        tool_data = tool_data.get("observations") or tool_data.get("tools") or [tool_data]
    if not isinstance(tool_data, list):
        return None

    reference_prediction = None
    for item in tool_data:
        if not isinstance(item, dict):
            continue
        is_model_tool = item.get("tool_name") == "model_auxiliary_tool" or item.get("name") == "model_auxiliary_tool"
        output = item.get("output", item)
        if is_model_tool and isinstance(output, dict):
            reference_prediction = output.get("reference_prediction") or output.get("ensemble_forecast_baseline")
            break
    if not isinstance(reference_prediction, list):
        return None
    return baseline_prediction_dict(reference_prediction, gt_dict)


def baseline_prediction_dict(values: list[float] | np.ndarray, gt_dict: dict[str, float]) -> dict[str, float] | None:
    timestamps = sorted(gt_dict)
    if len(timestamps) != len(values):
        return None
    try:
        return {timestamp: float(value) for timestamp, value in zip(timestamps, values)}
    except (TypeError, ValueError):
        return None


def extract_answer_from_tags(answer_text: str) -> str:
    if not answer_text:
        return ""
    matches = re.findall(r"<answer>(.*?)</answer>", answer_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return answer_text.strip()


def serialize_forecast(value: str | list[float] | np.ndarray | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    array = np.asarray(value, dtype=float)
    payload = {str(idx): float(item) for idx, item in enumerate(array.tolist())}
    return json.dumps(payload)
