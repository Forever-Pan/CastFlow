from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils.io import (
    DatasetView,
    load_dataset,
    make_windows,
    attach_covariate_forecasts,
    atomic_write_csv,
    forecast_timestamps,
)
from utils.prompting import ensure_prompt_config, prompt_identity, render_prompt


def generate_prompt_rows(
    data_path: str | Path,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    max_windows: int | None = None,
    covariate_forecasts_path: str | Path | None = None,
    covariate_issued_at: str | None = None,
    covariate_issue_col: str = "issued_at",
    prompt_config: Any = None,
) -> pd.DataFrame:
    dataset = load_dataset(
        data_path, target_col=target_col, timestamp_col=timestamp_col
    )
    covariates = (
        pd.read_csv(covariate_forecasts_path, keep_default_na=False)
        if covariate_forecasts_path
        else None
    )
    return windows_to_prompt_rows(
        dataset,
        lookback,
        horizon,
        stride=stride,
        max_windows=max_windows,
        covariate_forecasts=covariates,
        covariate_issued_at=covariate_issued_at,
        covariate_issue_col=covariate_issue_col,
        prompt_config=prompt_config,
    )


def windows_to_prompt_rows(
    dataset: DatasetView,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    covariate_forecasts: pd.DataFrame | None = None,
    covariate_issued_at: str | None = None,
    covariate_issue_col: str = "issued_at",
    prompt_config: Any = None,
) -> pd.DataFrame:
    prompt_config = ensure_prompt_config(prompt_config)
    identity = prompt_identity(prompt_config)
    windows = make_windows(
        dataset,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
    )
    rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        if window.future is None:
            continue
        input_json = series_to_json(
            window.history, window.target_col, window.timestamp_col
        )
        tool_context: str | dict[str, Any] = ""
        if covariate_forecasts is not None:
            attach_covariate_forecasts(
                window,
                covariate_forecasts,
                horizon,
                issued_at=covariate_issued_at,
                issue_col=covariate_issue_col,
            )
            tool_context = {
                "origin_issued_covariate_forecasts": json.loads(
                    window.covariate_forecasts.to_json(
                        orient="records", date_format="iso"
                    )
                )
            }
        rows.append(
            {
                "idx": idx,
                "input": input_json,
                "ground_truth": series_to_json(
                    window.future, window.target_col, window.timestamp_col
                ),
                "tool": "",
                "prompt": build_forecasting_prompt(
                    history=window.history,
                    target_col=window.target_col,
                    timestamp_col=window.timestamp_col,
                    horizon=horizon,
                    lookback=lookback,
                    tool=tool_context,
                    prompt_config=prompt_config,
                    expected_timestamps=forecast_timestamps(window, horizon),
                ),
                "system_prompt": render_prompt(prompt_config, "forecast_system"),
                "prompt_mode": identity["mode"],
                "prompt_fingerprint": identity["fingerprint"],
                "target_col": window.target_col,
                "lookback": lookback,
                "horizon": horizon,
            }
        )
    return pd.DataFrame(rows)


def build_forecasting_prompt(
    history: pd.DataFrame,
    target_col: str,
    timestamp_col: str | None,
    horizon: int,
    lookback: int | None = None,
    tool: str | dict[str, Any] = "",
    domain_description: str = "",
    prompt_config: Any = None,
    expected_timestamps: list[str] | None = None,
) -> str:
    """Render a user prompt from a private pack or sanitized reference template.

    Adapt the template to the task before using it for production training.
    Runtime values are serialized as data, never interpreted as template syntax.
    """
    prompt_config = ensure_prompt_config(prompt_config)
    context = {
        "role": "forecast",
        "target": target_col,
        "lookback": lookback or len(history),
        "horizon": horizon,
        "sampling_interval": infer_frequency_text(history, timestamp_col),
        "observed_history": json.loads(
            series_to_json(history, target_col, timestamp_col)
        ),
        "tool_evidence": tool,
        "task_context": domain_description,
        "output_contract": forecast_output_contract(horizon, expected_timestamps),
    }
    return render_prompt(
        prompt_config,
        "forecast_user",
        context=json.dumps(
            context, ensure_ascii=False, allow_nan=False, sort_keys=True
        ),
    )


def forecast_output_contract(
    horizon: int, timestamps: list[str] | None
) -> dict[str, Any]:
    keys = list(timestamps) if timestamps else [str(i) for i in range(horizon)]
    if len(keys) != horizon or len(set(keys)) != horizon:
        raise ValueError(
            "forecast timestamp count or uniqueness does not match the horizon"
        )
    return {
        "container": "answer",
        "encoding": "json_object",
        "keys": keys,
        "values": "finite_numbers",
        "length": horizon,
        "evidence_element": "tool_think",
    }


def build_api_forecasting_messages(
    history: pd.DataFrame,
    target_col: str,
    timestamp_col: str | None,
    horizon: int,
    tool_outputs: dict[str, Any],
    tool_schedule: list[str],
    lookback: int | None = None,
    retry_feedback: str = "",
    future: pd.DataFrame | None = None,
    retrieved_memory: list[dict[str, Any]] | None = None,
    expected_timestamps: list[str] | None = None,
    prompt_config: Any = None,
) -> tuple[str, str]:
    """Render neutral request data through the selected task-specific prompt pack."""
    prompt_config = ensure_prompt_config(prompt_config)
    tool_section = build_tool_section(tool_outputs, tool_schedule)
    exogenous_context = build_future_exogenous_context(
        future=future,
        target_col=target_col,
        timestamp_col=timestamp_col,
        tool_outputs=tool_outputs,
        horizon=horizon,
    )
    context = json.dumps(
        {
            "role": "forecast",
            "target": target_col,
            "lookback": lookback or len(history),
            "horizon": horizon,
            "sampling_interval": infer_frequency_text(history, timestamp_col),
            "observed_history": json.loads(
                series_to_json(history, target_col, timestamp_col)
            ),
            "tool_evidence": json.loads(tool_section),
            "issued_covariate_forecasts": (
                json.loads(exogenous_context) if exogenous_context else {}
            ),
            "historical_procedural_examples": procedural_guidance(
                retrieved_memory or []
            ),
            "previous_attempt_feedback": retry_feedback,
            "output_contract": forecast_output_contract(horizon, expected_timestamps),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )
    # Historical response delimiters are data, not current reasoning-channel controls.
    context = re.sub(
        r"</?think\s*>",
        lambda match: (
            "[historical reasoning]"
            if not match.group(0).startswith("</")
            else "[/historical reasoning]"
        ),
        context,
        flags=re.IGNORECASE,
    )
    return (
        render_prompt(prompt_config, "forecast_system"),
        render_prompt(prompt_config, "forecast_user", context=context),
    )


def build_tool_section(tool_outputs: dict[str, Any], tool_schedule: list[str]) -> str:
    blocks: dict[str, Any] = {}
    for name in tool_schedule:
        output: Any = tool_outputs.get(name, {})
        if name == "model_auxiliary_tool" and isinstance(output, dict):
            prior = output.get("reference_prediction")
            if prior is None:
                prior = output.get("ensemble_forecast_baseline")
            output = {"reference_prediction": prior}
        blocks[name] = {"description": tool_description(name), "output": output}
    return json.dumps(blocks, ensure_ascii=False, allow_nan=False, sort_keys=True)


def build_future_exogenous_context(
    future: pd.DataFrame | None,
    target_col: str,
    timestamp_col: str | None,
    tool_outputs: dict[str, Any],
    horizon: int,
) -> str:
    # This helper only formats auxiliary future values. Domain-specific meaning
    # should be added by downstream prompt customization when needed.
    if future is None or future.empty:
        return ""
    if target_col in future.columns:
        raise ValueError(
            "covariate context must be supplied separately from future target observations"
        )
    if len(future) != horizon:
        raise ValueError(
            "covariate forecasts must match the complete requested horizon"
        )
    excluded = {
        target_col,
        timestamp_col,
        "predicted_ans",
        "features_used",
        "ground_truth",
        "answer",
        "issued_at",
        "issue_time",
    }
    names = [name for name in future.columns if name not in excluded]
    exog_sequences: dict[str, list[float]] = {}
    exog_stats: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in excluded or name not in future.columns:
            continue
        values = (
            pd.to_numeric(future[name], errors="coerce")
            .head(horizon)
            .to_numpy(dtype=float)
        )
        if values.size != horizon or not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"covariate {name!r} must contain the complete finite forecast horizon"
            )
        rounded_values = [float(value) for value in values.tolist()]
        exog_sequences[str(name)] = rounded_values
        exog_stats[str(name)] = {
            "mean": round(float(values.mean()), 2),
            "std": round(float(values.std()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "trend": "increasing" if values[-1] > values[0] else "decreasing",
            "values": rounded_values,
        }
    if not exog_sequences:
        return ""
    return json.dumps(
        {
            "values": exog_sequences,
            "statistics": exog_stats,
            "horizon": horizon,
            "availability": "origin_issued_forecasts",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )


def insert_exogenous_context(tool_section: str, exogenous_context: str) -> str:
    """Merge serialized neutral evidence without adding private instructions."""
    return json.dumps(
        {
            "tools": json.loads(tool_section or "{}"),
            "covariate_forecasts": json.loads(exogenous_context or "{}"),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )


def tool_description(name: str) -> str:
    descriptions = {
        "model_auxiliary_tool": "Reference forecast.",
        "exogenous_analysis_tool": "Observed auxiliary-variable associations.",
        "statistical_analysis_tool": "Observed distribution summary.",
        "trend_analysis_tool": "Observed trend summary.",
        "basic_statistics_tool": "Observed temporal statistics.",
        "changepoint_trend_tool": "Observed change indicators.",
        "autoregressive_residual_tool": "Training-side residual summary.",
        "cross_channel_tool": "Observed cross-channel summary.",
        "data_quality_tool": "Input quality summary.",
        "event_summary_tool": "Observed variation summary.",
        "comprehensive_feature_tool": "Combined diagnostic data.",
    }
    return descriptions.get(name, "CastFlow diagnostic tool.")


def series_to_json(
    frame: pd.DataFrame, target_col: str, timestamp_col: str | None
) -> str:
    values = pd.to_numeric(frame[target_col], errors="coerce").astype(float)
    if timestamp_col and timestamp_col in frame.columns:
        timestamps = pd.to_datetime(frame[timestamp_col], errors="raise")
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
            raise ValueError(
                "series timestamps must be present and strictly increasing"
            )
        keys = timestamps.astype(str)
    else:
        keys = pd.Series([str(i) for i in range(len(frame))])
    if values.isna().any() or not all(math.isfinite(value) for value in values):
        raise ValueError("series contains missing or non-finite target values")
    if len(set(map(str, keys))) != len(values):
        raise ValueError("series timestamps must be unique")
    payload = {str(key): float(value) for key, value in zip(keys, values)}
    return json.dumps(payload, ensure_ascii=False)


def write_prompt_csv(
    data_path: str | Path,
    output_path: str | Path,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    max_windows: int | None = None,
    covariate_forecasts_path: str | Path | None = None,
    covariate_issued_at: str | None = None,
    covariate_issue_col: str = "issued_at",
    prompt_config: Any = None,
) -> str:
    if Path(data_path).resolve() == Path(output_path).resolve():
        raise ValueError("prompt output must differ from its source dataset")
    rows = generate_prompt_rows(
        data_path,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        target_col=target_col,
        timestamp_col=timestamp_col,
        max_windows=max_windows,
        covariate_forecasts_path=covariate_forecasts_path,
        covariate_issued_at=covariate_issued_at,
        covariate_issue_col=covariate_issue_col,
        prompt_config=prompt_config,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_path, rows)
    return str(output_path)


def infer_frequency_text(history: pd.DataFrame, timestamp_col: str | None) -> str:
    if not timestamp_col or timestamp_col not in history.columns or len(history) < 2:
        return "the original sampling interval"
    stamps = pd.to_datetime(history[timestamp_col], errors="coerce").dropna()
    if len(stamps) < 2:
        return "the original sampling interval"
    delta = stamps.iloc[-1] - stamps.iloc[-2]
    seconds = int(delta.total_seconds())
    if seconds % 86400 == 0:
        days = seconds // 86400
        return "1 day" if days == 1 else f"{days} days"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return "1 hour" if hours == 1 else f"{hours} hours"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    return str(delta)


def infer_task_text(target_col: str, columns: pd.Index) -> str:
    # Avoid baking benchmark-specific industry semantics into the public prompt.
    # Domain owners can replace this helper or pass stronger context separately.
    return f"predict the target variable `{target_col}`"


def procedural_guidance(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose archived procedures and examples with their historical scope."""
    guidance = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        schedule = entry.get("tool_schedule")
        if not isinstance(schedule, list):
            schedule = [
                call.get("tool_name", "")
                for call in entry.get("tool_calls", [])
                if isinstance(call, dict)
            ]
        guidance.append(
            {
                "scope": "historical_training_example",
                "dataset": entry.get("dataset_name", ""),
                "tool_schedule": schedule,
                "historical_diagnostics": entry.get(
                    "diagnostic_outputs", entry.get("tool_outputs", {})
                ),
                "historical_forecasting_response": entry.get(
                    "Forecasting_answer", entry.get("forecast_answer", "")
                ),
            }
        )
    return guidance
