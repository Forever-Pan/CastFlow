from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
from collections import Counter
from copy import deepcopy
import hashlib
import json
from threading import RLock
from pathlib import Path
from typing import Iterator
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.config import CastFlowConfig
from utils.schema import TimeSeriesWindow, ToolObservation
from anchorer import (
    AnchorLibrary,
    anchor_forecast_from_library,
    primitive_model_forecasts,
)


ToolFn = Callable[[TimeSeriesWindow, CastFlowConfig], dict[str, Any]]
_ANCHOR_LIBRARY_CACHE: OrderedDict[tuple[str, str], AnchorLibrary] = OrderedDict()
_ANCHOR_LIBRARY_LOCK = RLock()


@dataclass
class _DiagnosticScope:
    window: TimeSeriesWindow
    history: pd.DataFrame
    signature: tuple[Any, ...]
    policy: str
    eligible: set[str]
    primitives: dict[tuple[Any, ...], Any]


_DIAGNOSTIC_SCOPE: ContextVar[_DiagnosticScope | None] = ContextVar(
    "castflow_diagnostic_scope", default=None
)
_PRIMITIVES = {
    "statistical_analysis_tool": {"statistics"},
    "basic_statistics_tool": {"quantiles", "autocorrelation", "spectrum"},
    "data_quality_tool": {"quality", "statistics"},
    "trend_analysis_tool": {"slope", "statistics"},
    "changepoint_trend_tool": {"differences", "statistics"},
    "cross_channel_tool": {"exogenous_frame", "lag_correlations", "correlations"},
    "exogenous_analysis_tool": {"exogenous_frame", "correlations"},
    "event_summary_tool": {"slope", "differences", "statistics"},
    "autoregressive_residual_tool": {"ar_residuals"},
}
_PRIMITIVES["comprehensive_feature_tool"] = set().union(
    *(
        _PRIMITIVES[name]
        for name in (
            "statistical_analysis_tool",
            "basic_statistics_tool",
            "data_quality_tool",
            "trend_analysis_tool",
            "changepoint_trend_tool",
            "event_summary_tool",
        )
    )
)
_PRIMITIVE_COUNTS = {
    name: Counter({key: 1 for key in keys}) for name, keys in _PRIMITIVES.items()
}
_PRIMITIVE_COUNTS["comprehensive_feature_tool"] = sum(
    (
        _PRIMITIVE_COUNTS[name]
        for name in (
            "statistical_analysis_tool",
            "basic_statistics_tool",
            "data_quality_tool",
            "trend_analysis_tool",
            "changepoint_trend_tool",
            "event_summary_tool",
        )
    ),
    Counter(),
)


def _config_signature(config: CastFlowConfig) -> tuple[Any, ...]:
    return (
        config.lookback,
        config.horizon,
        config.seasonal_period,
        config.allow_train_only_tools,
    )


def _primitive(
    name: str,
    window: TimeSeriesWindow,
    config: CastFlowConfig,
    compute: Callable[[], Any],
) -> Any:
    scope = _DIAGNOSTIC_SCOPE.get()
    active = (
        scope is not None
        and scope.window is window
        and scope.history is window.history
        and scope.signature == _config_signature(config)
    )
    if not active or scope.policy == "none" or name not in scope.eligible:
        return compute()
    target = np.ascontiguousarray(window.target_history)
    digest = hashlib.sha256(target.tobytes()).hexdigest()
    extra = ""
    if name in {"exogenous_frame", "correlations", "lag_correlations"}:
        frame = window.history
        extra = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        ).hexdigest()
        extra += str(
            tuple(
                (str(col), str(dtype))
                for col, dtype in zip(frame.columns, frame.dtypes)
            )
        )
    key = (
        name,
        target.shape,
        target.dtype.str,
        digest,
        extra,
        _config_signature(config),
    )
    if key not in scope.primitives:
        value = compute()
        scope.primitives[key] = deepcopy(value)
        return value
    return deepcopy(scope.primitives[key])


def _statistics(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> tuple[float, float, float, float]:
    y = window.target_history
    return _primitive(
        "statistics",
        window,
        config,
        lambda: (
            float(np.mean(y)),
            float(np.std(y)),
            float(np.min(y)),
            float(np.max(y)),
        ),
    )


def _differences(window: TimeSeriesWindow, config: CastFlowConfig) -> np.ndarray:
    return _primitive(
        "differences", window, config, lambda: np.diff(window.target_history)
    )


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
        if not isinstance(spec.name, str) or not spec.name or not callable(spec.fn):
            raise ValueError("tool registration requires a name and callable")
        self._tools[spec.name] = spec

    @contextmanager
    def request_scope(
        self, window: TimeSeriesWindow, config: CastFlowConfig
    ) -> Iterator[None]:
        """Reuse pure diagnostics only within this request; model calls are excluded."""
        policy = getattr(config, "diagnostic_cache_policy", "current_plan")
        policy = {"independent": "none", "all_primitive": "all_primitives"}.get(
            policy, policy
        )
        if policy not in {"none", "all_primitives", "current_plan"}:
            raise ValueError(f"unknown diagnostic cache policy: {policy!r}")
        eligible = (
            set().union(*_PRIMITIVES.values()) if policy == "all_primitives" else set()
        )
        scope = _DiagnosticScope(
            window, window.history, _config_signature(config), policy, eligible, {}
        )
        token = _DIAGNOSTIC_SCOPE.set(scope)
        try:
            yield
        finally:
            scope.primitives.clear()
            _DIAGNOSTIC_SCOPE.reset(token)

    def set_plan(self, tool_schedule: list[str]) -> None:
        scope = _DIAGNOSTIC_SCOPE.get()
        if scope is not None and scope.policy == "current_plan":
            counts = sum(
                (_PRIMITIVE_COUNTS.get(name, Counter()) for name in tool_schedule),
                Counter(),
            )
            scope.eligible = {name for name, count in counts.items() if count > 1}
            scope.primitives = {
                key: value
                for key, value in scope.primitives.items()
                if key[0] in scope.eligible
            }

    def execute(
        self, name: str, window: TimeSeriesWindow, config: CastFlowConfig
    ) -> ToolObservation:
        if name not in self._tools:
            raise KeyError(f"unknown CastFlow tool: {name}")
        spec = self._tools[name]
        if spec.train_only and not config.allow_train_only_tools:
            return ToolObservation(
                tool_name=name,
                category=spec.category,
                output={"skipped": True, "reason": "train_only_tool_disabled"},
            )
        if (
            window.target_history.ndim != 1
            or len(window.target_history) == 0
            or not np.isfinite(window.target_history).all()
        ):
            raise ValueError("diagnostic target lookback must be nonempty and finite")
        output = spec.fn(window, config)
        _require_finite_output(output)
        return ToolObservation(tool_name=name, category=spec.category, output=output)

    def names(self) -> list[str]:
        return list(self._tools)

    def register_defaults(self) -> None:
        self.register(
            ToolSpec(
                "model_auxiliary_tool", "foundational_anchorer", model_auxiliary_tool
            )
        )
        self.register(
            ToolSpec(
                "statistical_analysis_tool",
                "statistical_spectral_profiler",
                statistical_analysis_tool,
            )
        )
        self.register(
            ToolSpec(
                "basic_statistics_tool",
                "statistical_spectral_profiler",
                basic_statistics_tool,
            )
        )
        self.register(
            ToolSpec(
                "data_quality_tool", "statistical_spectral_profiler", data_quality_tool
            )
        )
        self.register(
            ToolSpec(
                "comprehensive_feature_tool",
                "statistical_spectral_profiler",
                comprehensive_feature_tool,
            )
        )
        self.register(
            ToolSpec("trend_analysis_tool", "dynamics_monitor", trend_analysis_tool)
        )
        self.register(
            ToolSpec(
                "changepoint_trend_tool", "dynamics_monitor", changepoint_trend_tool
            )
        )
        self.register(
            ToolSpec("cross_channel_tool", "dynamics_monitor", cross_channel_tool)
        )
        self.register(
            ToolSpec(
                "exogenous_analysis_tool", "dynamics_monitor", exogenous_analysis_tool
            )
        )
        self.register(
            ToolSpec("event_summary_tool", "dynamics_monitor", event_summary_tool)
        )
        self.register(
            ToolSpec(
                "autoregressive_residual_tool",
                "residual_diagnoser",
                autoregressive_residual_tool,
                train_only=True,
            )
        )


def model_auxiliary_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    horizon = config.horizon
    seasonal = max(1, config.seasonal_period)
    if config.anchor_library_path:
        library = load_anchor_library(config.anchor_library_path)
        if library.target_col != window.target_col:
            raise ValueError(
                "anchor library target column differs from the current request"
            )
        if library.dataset_name and config.dataset_name:
            from utils.registry import normalize_dataset_name

            if (
                normalize_dataset_name(library.dataset_name).casefold()
                != normalize_dataset_name(config.dataset_name).casefold()
            ):
                raise ValueError(
                    "anchor library dataset differs from the current request"
                )
        timestamps = (
            window.history[window.timestamp_col] if window.timestamp_col else None
        )
        anchored = anchor_forecast_from_library(
            y,
            horizon,
            seasonal,
            library,
            top_k=max(1, config.memory_top_k),
            timestamps=timestamps,
        )
        anchored["source"] = "anchor_library"
        return anchored
    if config.use_api:
        raise ValueError(
            "an offline anchor library is required for model-backed forecasting; configure anchor_library_path"
        )

    forecasts = {
        name: values.tolist()
        for name, values in primitive_model_forecasts(y, horizon, seasonal).items()
    }
    weights = default_ensemble_weights(forecasts)
    baseline = np.zeros(horizon, dtype=float)
    for name, values in forecasts.items():
        baseline += weights[name] * np.asarray(values, dtype=float)

    return {
        "ensemble_forecast_baseline": round_list(baseline),
        "reference_prediction": round_list(baseline),
        "component_forecasts": {
            name: round_list(values) for name, values in forecasts.items()
        },
        "weights": {name: round(float(weight), 6) for name, weight in weights.items()},
        "source": "offline_primitive_ensemble",
        "fallback_reason": "API-disabled execution without an anchor library",
    }


def load_anchor_library(path: str) -> AnchorLibrary:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    key = (str(resolved), hashlib.sha256(payload).hexdigest())
    with _ANCHOR_LIBRARY_LOCK:
        if key not in _ANCHOR_LIBRARY_CACHE:
            library = AnchorLibrary.from_dict(json.loads(payload))
            for stale in [
                item for item in _ANCHOR_LIBRARY_CACHE if item[0] == str(resolved)
            ]:
                del _ANCHOR_LIBRARY_CACHE[stale]
            _ANCHOR_LIBRARY_CACHE[key] = library
            while len(_ANCHOR_LIBRARY_CACHE) > 16:
                _ANCHOR_LIBRARY_CACHE.popitem(last=False)
        _ANCHOR_LIBRARY_CACHE.move_to_end(key)
        return _ANCHOR_LIBRARY_CACHE[key]


def statistical_analysis_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    mean, std, minimum, maximum = _statistics(window, config)
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(minimum, 6),
        "max": round(maximum, 6),
        "range": round(maximum - minimum, 6),
    }


def basic_statistics_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    median, mad, q05, q95 = _primitive(
        "quantiles",
        window,
        config,
        lambda: (
            float(np.median(y)),
            float(np.median(np.abs(y - np.median(y)))),
            float(np.quantile(y, 0.05)),
            float(np.quantile(y, 0.95)),
        ),
    )
    acf1, acfs = _primitive(
        "autocorrelation",
        window,
        config,
        lambda: (
            autocorrelation(y, 1),
            autocorrelation(y, min(config.seasonal_period, max(1, len(y) - 1))),
        ),
    )
    entropy = _primitive("spectrum", window, config, lambda: spectral_entropy(y))
    return {
        "median": round(median, 6),
        "mad": round(mad, 6),
        "acf1": round(acf1, 6),
        "acf_s": round(acfs, 6),
        "spectral_entropy": round(entropy, 6),
        "q05": round(q05, 6),
        "q95": round(q95, 6),
    }


def data_quality_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    return dict(
        _primitive("quality", window, config, lambda: _data_quality(window, config))
    )


def _data_quality(window: TimeSeriesWindow, config: CastFlowConfig) -> dict[str, Any]:
    y = window.target_history
    missing_ratio = float(pd.Series(y).isna().mean())
    unique_ratio = float(len(np.unique(y)) / len(y)) if len(y) else 0.0
    mean, std, _, _ = _statistics(window, config)
    clipping_boundary = [mean - 3.0 * std, mean + 3.0 * std]
    return {
        "dropout_ratio": round(missing_ratio, 6),
        "constant_channel_ratio": 1.0 if np.ptp(y) == 0 else 0.0,
        "unique_ratio": round(unique_ratio, 6),
        "clipping_boundary": round_list(clipping_boundary),
    }


def comprehensive_feature_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    return {
        "statistical": statistical_analysis_tool(window, config),
        "basic": basic_statistics_tool(window, config),
        "trend": trend_analysis_tool(window, config),
        "changepoint": changepoint_trend_tool(window, config),
        "quality": data_quality_tool(window, config),
        "event": event_summary_tool(window, config),
    }


def trend_analysis_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    slope = _primitive("slope", window, config, lambda: linear_slope(y))
    _, _, minimum, maximum = _statistics(window, config)
    spread = maximum - minimum
    return {
        "slope": round(slope, 6),
        "trend_direction": (
            "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        ),
        "trend_strength": round(abs(slope) / spread, 6) if spread else 0.0,
    }


def changepoint_trend_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    diff = _differences(window, config)
    if diff.size == 0:
        return {"changepoint_count": 0, "changepoint_score": 0.0, "slope_max": 0.0}
    threshold = float(np.mean(np.abs(diff)) + 2.0 * np.std(diff))
    changepoints = np.where(np.abs(diff) > threshold)[0]
    second = np.diff(diff)
    return {
        "changepoint_count": int(len(changepoints)),
        "changepoint_score": round(
            float(np.max(np.abs(diff)) / (_statistics(window, config)[1] + 1e-8)), 6
        ),
        "slope_max": round(float(np.max(diff)), 6),
        "slope_min": round(float(np.min(diff)), 6),
        "slope_second_diff_max": round(
            float(np.max(second)) if second.size else 0.0, 6
        ),
        "recent_momentum": round(float(np.mean(diff[-min(5, len(diff)) :])), 6),
    }


def cross_channel_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    frame = _primitive(
        "exogenous_frame", window, config, lambda: numeric_exogenous_frame(window)
    )
    if frame.empty:
        return {"available": False, "reason": "no_exogenous_numeric_columns"}
    target = window.target_history
    best: dict[str, Any] | None = None
    correlations = dict(
        _primitive(
            "correlations",
            window,
            config,
            lambda: sorted(
                [
                    (col, safe_corr(target, frame[col].to_numpy(dtype=float)))
                    for col in frame.columns
                ],
                key=lambda item: abs(item[1]),
                reverse=True,
            ),
        )
    )
    relationships = _primitive(
        "lag_correlations",
        window,
        config,
        lambda: [
            (
                col,
                correlations[col],
                *best_lag_correlation(
                    target,
                    frame[col].to_numpy(dtype=float),
                    max_lag=min(8, len(target) // 4),
                ),
            )
            for col in frame.columns
        ],
    )
    for col, corr, lag, lag_corr in relationships:
        candidate = {
            "column": col,
            "correlation": corr,
            "lead_lag_shift": lag,
            "lead_lag_peak_corr": lag_corr,
        }
        if best is None or abs(candidate["lead_lag_peak_corr"]) > abs(
            best["lead_lag_peak_corr"]
        ):
            best = candidate
    return {"available": True, "strongest_relationship": best}


def exogenous_analysis_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    frame = _primitive(
        "exogenous_frame", window, config, lambda: numeric_exogenous_frame(window)
    )
    if frame.empty:
        return {"available": False, "top_correlated": []}
    target = window.target_history
    correlations = list(
        _primitive(
            "correlations",
            window,
            config,
            lambda: sorted(
                [
                    (col, safe_corr(target, frame[col].to_numpy(dtype=float)))
                    for col in frame.columns
                ],
                key=lambda item: abs(item[1]),
                reverse=True,
            ),
        )
    )
    result = {
        "available": True,
        "top_k_names": [col for col, _ in correlations[:3]],
        "correlations": {col: round(float(value), 6) for col, value in correlations},
        "top_correlated": [
            (col, round(float(value), 6)) for col, value in correlations[:3]
        ],
    }
    forecasts = getattr(window, "covariate_forecasts", None)
    if forecasts is not None:
        result["issued_forecast_columns"] = [
            str(col) for col in forecasts.columns if col in frame.columns
        ]
    return result


def event_summary_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    slope = _primitive("slope", window, config, lambda: linear_slope(y))
    changes = _differences(window, config)
    oscillation_ratio = (
        float(np.mean(np.sign(changes[1:]) != np.sign(changes[:-1])))
        if len(changes) > 1
        else 0.0
    )
    if oscillation_ratio > 0.55:
        pattern = "oscillation"
    elif abs(slope) <= max(1e-8, _statistics(window, config)[1] * 0.01):
        pattern = "flat"
    elif slope > 0:
        pattern = "rise"
    else:
        pattern = "fall"
    return {
        "dominant_pattern": pattern,
        "oscillation_ratio": round(oscillation_ratio, 6),
    }


def autoregressive_residual_tool(
    window: TimeSeriesWindow, config: CastFlowConfig
) -> dict[str, Any]:
    y = window.target_history
    order, resid = _primitive(
        "ar_residuals", window, config, lambda: fit_autoregressive_residuals(y)
    )
    scale = float(np.max(np.abs(resid))) if len(resid) else 0.0
    normalized = resid / scale if scale else np.zeros_like(resid)
    denominator = float(np.dot(normalized, normalized))
    acf = (
        float(np.dot(normalized[1:], normalized[:-1]) / denominator)
        if denominator
        else 0.0
    )
    return {
        "residual_mean": round(float(np.mean(resid)), 6),
        "residual_max": round(float(np.max(np.abs(resid))), 6),
        "residual_acf1": round(acf, 6),
        "autoregressive_order": order,
    }


def fit_autoregressive_residuals(y: np.ndarray) -> tuple[int, np.ndarray]:
    """Choose lag order by AIC on a common sample, then refit the lookback."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("autoregressive fitting requires finite lookback values")
    if np.ptp(values) == 0:
        return 0, np.zeros_like(values)
    scale = max(float(np.max(np.abs(values))), 1.0)
    normalized = values / scale
    max_lag = min(10, max(0, (len(values) - 2) // 3))
    holdback = max_lag
    target = normalized[holdback:]
    best_order, best_aic = 0, float("inf")
    for order in range(max_lag + 1):
        design = np.column_stack(
            [np.ones(len(target))]
            + [
                normalized[holdback - lag : len(values) - lag]
                for lag in range(1, order + 1)
            ]
        )
        coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < order + 1:
            continue
        residual = target - design @ coef
        variance = max(
            float(np.dot(residual, residual) / len(target)), np.finfo(float).tiny
        )
        aic = len(target) * np.log(variance) + 2 * (order + 2)
        if aic < best_aic:
            best_order, best_aic = order, aic
    target = normalized[best_order:]
    design = np.column_stack(
        [np.ones(len(target))]
        + [
            normalized[best_order - lag : len(values) - lag]
            for lag in range(1, best_order + 1)
        ]
    )
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = (target - design @ coef) * scale
    if float(np.max(np.abs(residual))) <= 64 * np.finfo(float).eps * scale:
        residual = np.zeros_like(residual)
    return best_order, residual


def numeric_exogenous_frame(window: TimeSeriesWindow) -> pd.DataFrame:
    excluded = {
        window.target_col,
        window.timestamp_col,
        "predicted_ans",
        "features_used",
    }
    cols = [col for col in window.history.columns if col not in excluded]
    numeric = {}
    for col in cols:
        values = pd.to_numeric(window.history[col], errors="coerce")
        if values.notna().sum() > 0:
            numeric[col] = values.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(numeric, index=window.history.index)


def default_ensemble_weights(forecasts: dict[str, list[float]]) -> dict[str, float]:
    if not forecasts:
        raise ValueError("offline expert pool is empty")
    return {name: 1 / len(forecasts) for name in forecasts}


def seasonal_naive_forecast(
    y: np.ndarray, horizon: int, seasonal_period: int
) -> np.ndarray:
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
    centered = x - np.mean(x)
    return float(
        np.dot(centered, np.asarray(y, dtype=float) - np.mean(y))
        / np.dot(centered, centered)
    )


def autocorrelation(y: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(y) <= lag:
        return 0.0
    left = y[:-lag]
    right = y[lag:]
    return safe_corr(left, right)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("correlation inputs must be one-dimensional and equally sized")
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2:
        return 0.0
    left_scale, right_scale = float(np.max(np.abs(left))), float(np.max(np.abs(right)))
    left = left / (left_scale or 1.0)
    right = right / (right_scale or 1.0)
    left, right = left - np.mean(left), right - np.mean(right)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(np.dot(left, right) / denom, -1, 1)) if denom > 0 else 0.0


def best_lag_correlation(
    target: np.ndarray, aux: np.ndarray, max_lag: int
) -> tuple[int, float]:
    best_lag = 0
    best_corr = safe_corr(target, aux)
    max_lag = min(max_lag, max(0, len(target) - 2))
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            corr = safe_corr(target[:-lag], aux[lag:])
        else:
            corr = safe_corr(target[-lag:], aux[:lag])
        if abs(corr) > abs(best_corr):
            best_lag = lag
            best_corr = corr
    return best_lag, round(float(best_corr), 6)


def spectral_entropy(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("spectral entropy requires a finite nonempty sequence")
    scale = float(np.max(np.abs(y)))
    y = y / (scale or 1.0)
    power = np.abs(np.fft.rfft(y)[1:]) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    probs = power / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return max(0.0, entropy)


def _require_finite_output(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_output(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_output(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("tool produced a nonfinite diagnostic")


def round_list(values: list[float] | np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]
