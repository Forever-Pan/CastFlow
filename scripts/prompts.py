from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import DatasetView, load_dataset, make_windows


SYSTEM_PROMPT = "You are a time series forecasting expert."


def generate_prompt_rows(
    data_path: str | Path,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    max_windows: int | None = None,
) -> pd.DataFrame:
    dataset = load_dataset(data_path, target_col=target_col, timestamp_col=timestamp_col)
    return windows_to_prompt_rows(dataset, lookback, horizon, stride=stride, max_windows=max_windows)


def windows_to_prompt_rows(
    dataset: DatasetView,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
) -> pd.DataFrame:
    windows = make_windows(dataset, lookback=lookback, horizon=horizon, stride=stride, max_windows=max_windows)
    rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        if window.future is None:
            continue
        input_json = series_to_json(window.history, window.target_col, window.timestamp_col)
        rows.append(
            {
                "idx": idx,
                "input": input_json,
                "ground_truth": series_to_json(window.future, window.target_col, window.timestamp_col),
                "tool": "",
                "prompt": build_forecasting_prompt(
                    history=window.history,
                    target_col=window.target_col,
                    timestamp_col=window.timestamp_col,
                    horizon=horizon,
                    lookback=lookback,
                    tool="",
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
    tool_text = tool if isinstance(tool, str) else json.dumps(tool, ensure_ascii=False)
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
Please think before you answer. Place your thinking process inside <think></think>.
During the thinking process, you need to consider the information provided by the tools (if any);
place this specific information under <tool_think></tool_think> within the thinking process.
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
    if exogenous_context and tool_section:
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

    user_message = "\n".join(
        part
        for part in [
            "## Task",
            f"Predict the next {horizon} time points based on the historical time series of {lookback} time points.",
            f"Timestamps increment by {frequency_text}.",
            "",
            "## Output Format Requirements",
            "**CRITICAL:**",
            f"1. You MUST predict EXACTLY {horizon} time points (no more, no less)",
            f"2. Output format: JSON object with exactly {horizon} key-value pairs",
            "3. Each key: timestamp string, each value: float number",
            "4. Timestamps continue from the last input timestamp with the same increment as the input",
            "5. Place your thinking in `<think></think>` tags",
            "6. Place your prediction in `<answer></answer>` tags",
            retry_context,
            "## Input Data",
            "",
            "### Historical Time Series:",
            "```json",
            input_json,
            "```",
            "",
            "## Tool Results (called tools only; ordered: mandatory tools first, then optional tools)",
            tool_section if tool_section else "No tools were called.",
            "",
            "## Instructions",
            "**CRITICAL: How to use tool results (MUST FOLLOW):**",
            "",
            "1. **Base Prediction (model_auxiliary_tool) - PRIMARY FOUNDATION:**",
            "   - The reference_prediction from model_auxiliary_tool is your PRIMARY baseline and MUST be used as the foundation.",
            "   - This reference_prediction already contains the correct temporal patterns and volatility from similar historical cases.",
            "   - DO NOT ignore or significantly deviate from reference_prediction. Your final prediction should closely follow its pattern and volatility.",
            "   - If reference_prediction shows fluctuations, your prediction MUST also show similar fluctuations.",
            "",
            "2. **Key Refinement (exogenous_analysis_tool / original exogenous_analysis):**",
            "   - Apply any provided auxiliary-signal analysis to refine the base prediction.",
            "   - Use the future-window auxiliary values only when they are available and relevant to the domain.",
            "",
            "3. **Auxiliary Analysis (statistical_analysis_tool, trend_analysis_tool):**",
            "   - These tools are for understanding data characteristics ONLY, NOT for direct prediction.",
            "   - **CRITICAL WARNING**: Do NOT simply extrapolate based on trend_direction (e.g., 'increasing').",
            "   - Time series often have fluctuations even when the overall trend is increasing or decreasing.",
            "   - The reference_prediction already captures the correct pattern - use it as your primary guide.",
            "",
            "4. **Advanced Feature Tools (LLM Correction Framework) - META-REASONING:**",
            "   If any of the following tools are called, use their outputs for intelligent prediction correction:",
            "",
            "   - **basic_statistics_tool**: Check `_llm_hints` for correction guidance.",
            "     * If median differs significantly from model prediction mean, apply BIAS CORRECTION (shift prediction curve).",
            "     * If mad is high, EXPAND confidence interval / uncertainty bounds.",
            "     * If acf_s is high but reference_prediction is flat, INJECT periodic fluctuations.",
            "     * If spec_entropy is high (noisy signal), use CONSERVATIVE strategy (e.g., moving average).",
            "",
            "   - **changepoint_trend_tool**: This is the CORE correction tool for fixing model lag.",
            "     * If changepoint_score is high, TRUNCATE memory - ignore pre-changepoint history, extrapolate from post-changepoint trend only.",
            "     * If slope_max is high in uptrend, BOOST prediction values with linear compensation.",
            "     * If slope_second_diff_max turns negative (decelerating), PREDICT REVERSAL early - lower the prediction curve.",
            "     * If monotone_duration is extremely long, consider MEAN REVERSION.",
            "     * If flatline_ratio is high, mark prediction as INVALID or output constant value.",
            "",
            "   - **autoregressive_residual_tool / original ar_residual_tool**: Use for residual-based correction.",
            "     * If residual_mean != 0, ADD residual mean back to prediction.",
            "     * If residual_acf1 is high, apply NONLINEAR correction using your reasoning.",
            "",
            "   - **cross_channel_tool**: Use for multi-variate correction.",
            "     * If lead_lag_shift_mean shows auxiliary variable leads by N steps, FORCE current variable to follow in next N steps.",
            "     * If co_anom_ratio is high (macro event detected), apply GLOBAL directional adjustment.",
            "",
            "   - **data_quality_tool**: This is the MOST IMPORTANT warning signal.",
            "     * If quality_dropout_ratio is high, EXPAND confidence interval significantly.",
            "     * If quality_saturation_ratio is high, CLIP predictions to historical saturation bounds.",
            "",
            "   - **event_summary_tool**: Use for semantic constraint.",
            "     * Check dominant_pattern_name and apply corresponding constraint:",
            "       - 'rise': maintain upward trend",
            "       - 'fall': maintain downward trend",
            "       - 'flat': keep prediction oscillating around mean",
            "       - 'oscillation': suppress unidirectional trends",
            "",
            "   - **comprehensive_feature_tool**: Contains ALL features. Check `_llm_comprehensive_hints` for guidance.",
            "",
            "**IMPORTANT REMINDERS:**",
            "- Your prediction MUST maintain similar volatility and fluctuation patterns as the reference_prediction.",
            "- Do NOT create a monotonically increasing or decreasing sequence unless the reference_prediction shows such a pattern.",
            "- If you use tool information, include it under `<tool_think></tool_think>` within `<think></think>`.",
            f"- Output ONLY one `<answer></answer>` JSON with exactly {horizon} points.",
        ]
        if part
    )
    return SYSTEM_PROMPT, user_message


def build_tool_section(tool_outputs: dict[str, Any], tool_schedule: list[str]) -> str:
    blocks: list[str] = []
    for name in tool_schedule:
        output: Any = tool_outputs.get(name, {})
        if name == "model_auxiliary_tool" and isinstance(output, dict):
            output = {
                "reference_prediction": output.get("reference_prediction")
                or output.get("ensemble_forecast_baseline")
            }
        blocks.append(
            "\n".join(
                [
                    f"### {name}",
                    f"- Description: {tool_description(name)}",
                    "",
                    "```json",
                    json.dumps(output, ensure_ascii=False, indent=2),
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
    exog_output = tool_outputs.get("exogenous_analysis_tool") or tool_outputs.get("exogenous_analysis") or {}
    if not isinstance(exog_output, dict) or exog_output.get("error"):
        return ""
    names = exog_output.get("top_k_names")
    if not names:
        top_correlated = exog_output.get("top_correlated", [])
        names = [item[0] for item in top_correlated if isinstance(item, (list, tuple)) and item]
    if not names:
        correlations = exog_output.get("correlations", {})
        if isinstance(correlations, dict):
            names = list(correlations)[:3]

    excluded = {target_col, timestamp_col, "predicted_ans", "features_used"}
    exog_sequences: dict[str, list[float]] = {}
    exog_stats: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in excluded or name not in future.columns:
            continue
        values = pd.to_numeric(future[name], errors="coerce").dropna().head(horizon).to_numpy(dtype=float)
        if values.size == 0:
            continue
        rounded_values = [round(float(value), 2) for value in values.tolist()]
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
            "**Exogenous future values for prediction (values only):**",
            f"- Correlations were computed on HISTORICAL window; sequences below are FUTURE window aligned to the {horizon}-step forecast horizon.",
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
        return tool_section[:insert_at] + "\n\n" + exogenous_context + "\n" + tool_section[insert_at:]
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


def series_to_json(frame: pd.DataFrame, target_col: str, timestamp_col: str | None) -> str:
    values = pd.to_numeric(frame[target_col], errors="coerce").astype(float)
    if timestamp_col and timestamp_col in frame.columns:
        keys = pd.to_datetime(frame[timestamp_col], errors="coerce").astype(str)
    else:
        keys = pd.Series([str(i) for i in range(len(frame))])
    payload = {str(key): round(float(value), 6) for key, value in zip(keys, values)}
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
) -> str:
    rows = generate_prompt_rows(
        data_path,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        target_col=target_col,
        timestamp_col=timestamp_col,
        max_windows=max_windows,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
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
