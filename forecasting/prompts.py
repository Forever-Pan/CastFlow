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
)


SYSTEM_PROMPT = "You are a time series forecasting expert."


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
) -> pd.DataFrame:
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
                ),
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
) -> str:
    """Build a prompt close to the original CastFlow SFT/RL prompt."""
    # Keep the base prompt domain-agnostic. Projects with strong industry priors
    # should inject them through `domain_description` rather than hard-coding them here.
    lookback = lookback or len(history)
    input_json = series_to_json(history, target_col, timestamp_col)
    tool_text = (
        tool
        if isinstance(tool, str)
        else json.dumps(tool, ensure_ascii=False, sort_keys=True)
    )
    frequency_text = infer_frequency_text(history, timestamp_col)
    task_text = infer_task_text(target_col, history.columns)
    return f"""
You are a time series forecasting expert.
Your current task is to {task_text}.
{domain_description}
Please predict the upcoming {horizon} time points based on the historical time series of {lookback} time points.
The timestamps increment by {frequency_text}.
The current input segment of the historical time series is as follows: {input_json}.
The tool invocation type and tool output results are as follows: {tool_text}.
Use the model reasoning channel to analyze the history before producing the final answer.
During the thinking process, you need to consider the information provided by the tools (if any);
record the relevant tool evidence in a tool_think element within the reasoning.
Place the results inside <answer></answer>; the format must contain exactly {horizon} time points.
""".strip()


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
) -> tuple[str, str]:
    """Build API messages using the detailed CastFlow forecasting prompt style."""
    input_json = series_to_json(history, target_col, timestamp_col)
    tool_section = build_tool_section(tool_outputs, tool_schedule)
    exogenous_context = build_future_exogenous_context(
        future=future,
        target_col=target_col,
        timestamp_col=timestamp_col,
        tool_outputs=tool_outputs,
        horizon=horizon,
    )
    if exogenous_context:
        tool_section = insert_exogenous_context(tool_section, exogenous_context)
    frequency_text = infer_frequency_text(history, timestamp_col)
    lookback = lookback or len(history)
    retry_context = ""
    if retry_feedback:
        retry_context = f"""

## Previous Attempt Feedback
The previous attempt had issues. Please fix all issues below:
{retry_feedback}
"""

    keys = expected_timestamps or [str(i) for i in range(horizon)]
    if len(keys) != horizon:
        raise ValueError("forecast timestamp count does not match the horizon")
    user_message = "\n".join(
        [
            "## Task",
            f"Generate a complete {horizon}-point numerical forecast from the observed {lookback}-point lookback.",
            f"Sampling interval: {frequency_text}.",
            "## Output contract",
            f"Return one final <answer> JSON object </answer> with exactly {horizon} finite numeric values.",
            "Use these keys in this order: " + json.dumps(keys, ensure_ascii=False),
            "Use the model's reasoning channel when available. Tool-based reasoning may be recorded in a tool_think element.",
            "Do not place sample answers inside the final answer element. Do not output arithmetic expressions, missing values, or additional forecast points.",
            "## Observed history",
            input_json,
            "## Current request tool evidence",
            tool_section or "No current tool evidence is available.",
            "## Historical procedural examples (training windows only)",
            json.dumps(
                procedural_guidance(retrieved_memory or []),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            "Archived values and answers describe past training instances. They are guidance for using tools, not observations or targets for this request.",
            retry_context,
            "## Forecasting instructions",
            "Use the ensemble forecasting prior as the numerical starting point. Interpret the current diagnostics and any supplied origin-issued covariate forecasts when deciding whether and how to refine it.",
            "Preserve or adjust the prior only as supported by the current evidence. Diagnostic statistics describe the observations; they do not prescribe a fixed correction formula or guarantee forecast accuracy.",
            "Generate the complete final numerical forecast. Do not assume that qualitative trend, correlation, or historic residual signals identify unknown future errors.",
        ]
    )

    user_message = re.sub(
        r"</?think\s*>",
        lambda match: (
            "[historical reasoning]"
            if not match.group(0).startswith("</")
            else "[/historical reasoning]"
        ),
        user_message,
        flags=re.IGNORECASE,
    )
    return SYSTEM_PROMPT, user_message


def build_tool_section(tool_outputs: dict[str, Any], tool_schedule: list[str]) -> str:
    blocks: list[str] = []
    for name in tool_schedule:
        output: Any = tool_outputs.get(name, {})
        if name == "model_auxiliary_tool" and isinstance(output, dict):
            prior = output.get("reference_prediction")
            if prior is None:
                prior = output.get("ensemble_forecast_baseline")
            output = {"reference_prediction": prior}
        blocks.append(
            "\n".join(
                [
                    f"### {name}",
                    f"- Description: {tool_description(name)}",
                    "",
                    "```json",
                    json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                ]
            )
        )
    return "\n\n".join(blocks)


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
    return "\n".join(
        [
            "**Origin-issued covariate forecasts for the prediction horizon:**",
            f"- Correlations describe the observed lookback. These separately supplied forecasts are aligned to the {horizon}-step horizon and were issued no later than the forecast origin.",
            "",
            "```json",
            json.dumps(exog_sequences, ensure_ascii=False, separators=(",", ":")),
            "```",
            "",
            "**Exogenous stats (future window):**",
            "",
            "```json",
            json.dumps(exog_stats, ensure_ascii=False, separators=(",", ":")),
            "```",
        ]
    )


def insert_exogenous_context(tool_section: str, exogenous_context: str) -> str:
    for marker in ("### exogenous_analysis_tool", "### exogenous_analysis"):
        start = tool_section.find(marker)
        if start == -1:
            continue
        next_block = tool_section.find("\n### ", start + len(marker))
        insert_at = next_block if next_block != -1 else len(tool_section)
        return (
            tool_section[:insert_at]
            + "\n\n"
            + exogenous_context
            + "\n"
            + tool_section[insert_at:]
        )
    return tool_section + "\n\n" + exogenous_context


def tool_description(name: str) -> str:
    descriptions = {
        "model_auxiliary_tool": "Base prediction tool: generates reference_prediction as the primary ensemble forecast baseline.",
        "exogenous_analysis_tool": "Analyzes correlations between exogenous variables and the target variable.",
        "statistical_analysis_tool": "Calculates mean, std, min, max, and range for validating prediction boundaries.",
        "trend_analysis_tool": "Analyzes trend direction and trend strength.",
        "basic_statistics_tool": "Extracts median, MAD, ACF, spectral entropy, and quantiles.",
        "changepoint_trend_tool": "Detects structural breaks, momentum, and recent trend changes.",
        "autoregressive_residual_tool": "Analyzes autoregressive residuals for systematic bias and nonlinear patterns.",
        "cross_channel_tool": "Analyzes cross-channel correlation and lead-lag relationships.",
        "data_quality_tool": "Assesses missingness, constant channels, and clipping boundaries.",
        "event_summary_tool": "Identifies dominant rise/fall/flat/oscillation pattern.",
        "comprehensive_feature_tool": "Aggregates multi-view diagnostic evidence.",
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
) -> str:
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
