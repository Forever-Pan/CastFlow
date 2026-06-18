from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.config import CastFlowConfig
from utils.schema import TimeSeriesWindow, ToolObservation
from anchorer import AnchorLibrary, anchor_forecast_from_library, primitive_model_forecasts


ToolFn = Callable[[TimeSeriesWindow, CastFlowConfig], dict[str, Any]]
_ANCHOR_LIBRARY_CACHE: dict[str, AnchorLibrary] = {}


@dataclass(slots=True)
class ToolSpec:
    name: str
    category: str
    fn: ToolFn
    train_only: bool = False


class MultiViewToolkit:
    """Paper-aligned toolkit containing four categories and eleven tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.register_defaults()

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def execute(self, name: str, window: TimeSeriesWindow, config: CastFlowConfig) -> ToolObservation:
        if name not in self._tools:
            raise KeyError(f"unknown CastFlow tool: {name}")
        spec = self._tools[name]
        if spec.train_only and not config.allow_train_only_tools:
            return ToolObservation(
                tool_name=name,
                category=spec.category,
                output={"skipped": True, "reason": "train_only_tool_disabled"},
            )
        return ToolObservation(tool_name=name, category=spec.category, output=spec.fn(window, config))

    def names(self) -> list[str]:
        return list(self._tools)

    def register_defaults(self) -> None:
        self.register(ToolSpec("model_auxiliary_tool", "foundational_anchorer", model_auxiliary_tool))
        self.register(ToolSpec("statistical_analysis_tool", "statistical_spectral_profiler", statistical_analysis_tool))
        self.register(ToolSpec("basic_statistics_tool", "statistical_spectral_profiler", basic_statistics_tool))
        self.register(ToolSpec("data_quality_tool", "statistical_spectral_profiler", data_quality_tool))
        self.register(ToolSpec("comprehensive_feature_tool", "statistical_spectral_profiler", comprehensive_feature_tool))
        self.register(ToolSpec("trend_analysis_tool", "dynamics_monitor", trend_analysis_tool))
        self.register(ToolSpec("changepoint_trend_tool", "dynamics_monitor", changepoint_trend_tool))
        self.register(ToolSpec("cross_channel_tool", "dynamics_monitor", cross_channel_tool))
        self.register(ToolSpec("exogenous_analysis_tool", "dynamics_monitor", exogenous_analysis_tool))
        self.register(ToolSpec("event_summary_tool", "dynamics_monitor", event_summary_tool))
        self.register(ToolSpec("autoregressive_residual_tool", "residual_diagnoser", autoregressive_residual_tool, train_only=True))


def model_auxiliary_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    horizon = config.horizon
    seasonal = max(1, config.seasonal_period)
    if config.anchor_library_path:
        try:
            library = load_anchor_library(config.anchor_library_path)
            timestamps = window.history[window.timestamp_col] if window.timestamp_col else None
            anchored = anchor_forecast_from_library(
                y,
                horizon,
                seasonal,
                library,
                top_k=max(1, config.memory_top_k),
                timestamps=timestamps,
            )
            anchored["source"] = "anchor_library"
            anchored["anchor_library_path"] = config.anchor_library_path
            return anchored
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "anchor_library_not_configured"

    forecasts = {name: values.tolist() for name, values in primitive_model_forecasts(y, horizon, seasonal).items()}
    weights = default_ensemble_weights(forecasts)
    baseline = np.zeros(horizon, dtype=float)
    for name, values in forecasts.items():
        baseline += weights[name] * np.asarray(values, dtype=float)

    return {
        "ensemble_forecast_baseline": round_list(baseline),
        "reference_prediction": round_list(baseline),
        "component_forecasts": {name: round_list(values) for name, values in forecasts.items()},
        "weights": {name: round(float(weight), 6) for name, weight in weights.items()},
        "source": "deterministic_fallback",
        "fallback_reason": fallback_reason,
    }


def load_anchor_library(path: str) -> AnchorLibrary:
    if path not in _ANCHOR_LIBRARY_CACHE:
        _ANCHOR_LIBRARY_CACHE[path] = AnchorLibrary.load(path)
    return _ANCHOR_LIBRARY_CACHE[path]


def statistical_analysis_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    return {
        "mean": round(float(np.mean(y)), 6),
        "std": round(float(np.std(y)), 6),
        "min": round(float(np.min(y)), 6),
        "max": round(float(np.max(y)), 6),
        "range": round(float(np.max(y) - np.min(y)), 6),
    }


def basic_statistics_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    acf1 = autocorrelation(y, lag=1)
    acfs = autocorrelation(y, lag=min(config.seasonal_period, max(1, len(y) - 1)))
    entropy = spectral_entropy(y)
    return {
        "median": round(median, 6),
        "mad": round(mad, 6),
        "acf1": round(acf1, 6),
        "acf_s": round(acfs, 6),
        "spectral_entropy": round(entropy, 6),
        "q05": round(float(np.quantile(y, 0.05)), 6),
        "q95": round(float(np.quantile(y, 0.95)), 6),
    }


def data_quality_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    missing_ratio = float(pd.Series(y).isna().mean())
    unique_ratio = float(len(np.unique(y)) / len(y)) if len(y) else 0.0
    std = float(np.std(y))
    mean = float(np.mean(y))
    clipping_boundary = [mean - 3.0 * std, mean + 3.0 * std]
    return {
        "dropout_ratio": round(missing_ratio, 6),
        "constant_channel_ratio": 1.0 if unique_ratio <= 0.02 else 0.0,
        "unique_ratio": round(unique_ratio, 6),
        "clipping_boundary": round_list(clipping_boundary),
    }


def comprehensive_feature_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    return {
        "statistical": statistical_analysis_tool(window, config),
        "basic": basic_statistics_tool(window, config),
        "trend": trend_analysis_tool(window, config),
        "changepoint": changepoint_trend_tool(window, config),
        "quality": data_quality_tool(window, config),
        "event": event_summary_tool(window, config),
    }


def trend_analysis_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    slope = linear_slope(y)
    spread = float(np.max(y) - np.min(y))
    return {
        "slope": round(slope, 6),
        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
        "trend_strength": round(abs(slope) / spread, 6) if spread else 0.0,
    }


def changepoint_trend_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    diff = np.diff(y)
    if diff.size == 0:
        return {"changepoint_count": 0, "changepoint_score": 0.0, "slope_max": 0.0}
    threshold = float(np.mean(np.abs(diff)) + 2.0 * np.std(diff))
    changepoints = np.where(np.abs(diff) > threshold)[0]
    second = np.diff(diff)
    return {
        "changepoint_count": int(len(changepoints)),
        "changepoint_score": round(float(np.max(np.abs(diff)) / (np.std(y) + 1e-8)), 6),
        "slope_max": round(float(np.max(diff)), 6),
        "slope_min": round(float(np.min(diff)), 6),
        "slope_second_diff_max": round(float(np.max(second)) if second.size else 0.0, 6),
        "recent_momentum": round(float(np.mean(diff[-min(5, len(diff)) :])), 6),
    }


def cross_channel_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    frame = numeric_exogenous_frame(window)
    if frame.empty:
        return {"available": False, "reason": "no_exogenous_numeric_columns"}
    target = window.target_history
    best: dict[str, Any] | None = None
    for col in frame.columns:
        values = frame[col].to_numpy(dtype=float)
        corr = safe_corr(target, values)
        lag, lag_corr = best_lag_correlation(target, values, max_lag=min(8, len(target) // 4))
        candidate = {"column": col, "correlation": corr, "lead_lag_shift": lag, "lead_lag_peak_corr": lag_corr}
        if best is None or abs(candidate["lead_lag_peak_corr"]) > abs(best["lead_lag_peak_corr"]):
            best = candidate
    return {"available": True, "strongest_relationship": best}


def exogenous_analysis_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    frame = numeric_exogenous_frame(window)
    if frame.empty:
        return {"available": False, "top_correlated": []}
    target = window.target_history
    correlations = [(col, safe_corr(target, frame[col].to_numpy(dtype=float))) for col in frame.columns]
    correlations.sort(key=lambda item: abs(item[1]), reverse=True)
    return {
        "available": True,
        "top_k_names": [col for col, _ in correlations[:3]],
        "correlations": {col: round(float(value), 6) for col, value in correlations},
        "top_correlated": [(col, round(float(value), 6)) for col, value in correlations[:3]],
    }


def event_summary_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    slope = linear_slope(y)
    changes = np.diff(y)
    oscillation_ratio = float(np.mean(np.sign(changes[1:]) != np.sign(changes[:-1]))) if len(changes) > 1 else 0.0
    if oscillation_ratio > 0.55:
        pattern = "oscillation"
    elif abs(slope) <= max(1e-8, np.std(y) * 0.01):
        pattern = "flat"
    elif slope > 0:
        pattern = "rise"
    else:
        pattern = "fall"
    return {"dominant_pattern": pattern, "oscillation_ratio": round(oscillation_ratio, 6)}


def autoregressive_residual_tool(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    if len(y) < 3:
        return {"residual_mean": 0.0, "residual_acf1": 0.0}
    pred = np.roll(y, 1)[1:]
    resid = y[1:] - pred
    return {
        "residual_mean": round(float(np.mean(resid)), 6),
        "residual_max": round(float(np.max(np.abs(resid))), 6),
        "residual_acf1": round(autocorrelation(resid, lag=1), 6),
    }


def numeric_exogenous_frame(window: TimeSeriesWindow) -> pd.DataFrame:
    excluded = {window.target_col, window.timestamp_col, "predicted_ans", "features_used"}
    cols = [col for col in window.history.columns if col not in excluded]
    numeric = {}
    for col in cols:
        values = pd.to_numeric(window.history[col], errors="coerce")
        if values.notna().sum() > 0:
            numeric[col] = values.ffill().bfill().fillna(0.0)
    return pd.DataFrame(numeric)


def default_ensemble_weights(forecasts: dict[str, list[float]]) -> dict[str, float]:
    base = {name: 1.0 for name in forecasts}
    if "seasonal_naive" in base:
        base["seasonal_naive"] = 1.6
    if "linear_drift" in base:
        base["linear_drift"] = 1.2
    total = sum(base.values())
    return {name: weight / total for name, weight in base.items()}


def seasonal_naive_forecast(y: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    tail = y[-seasonal_period:]
    repeats = int(np.ceil(horizon / len(tail)))
    return np.tile(tail, repeats)[:horizon].astype(float)


def linear_drift_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    slope = linear_slope(y[-min(len(y), 24) :])
    steps = np.arange(1, horizon + 1, dtype=float)
    return y[-1] + slope * steps


def linear_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y.astype(float), 1)[0])


def autocorrelation(y: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(y) <= lag:
        return 0.0
    left = y[:-lag]
    right = y[lag:]
    return safe_corr(left, right)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def best_lag_correlation(target: np.ndarray, aux: np.ndarray, max_lag: int) -> tuple[int, float]:
    best_lag = 0
    best_corr = safe_corr(target, aux)
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            corr = safe_corr(target[lag:], aux[:-lag])
        else:
            corr = safe_corr(target[:lag], aux[-lag:])
        if abs(corr) > abs(best_corr):
            best_lag = lag
            best_corr = corr
    return best_lag, round(float(best_corr), 6)


def spectral_entropy(y: np.ndarray) -> float:
    centered = y - np.mean(y)
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    probs = power / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(len(probs))) if len(probs) > 1 else 0.0


def round_list(values: list[float] | np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]
