from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from utils.io import load_dataset, make_windows
from utils.schema import TimeSeriesWindow
from anchorer.models import forecast_with_anchorer_models
from utils.metrics import mse


MODEL_NAMES = ("SeasonalNaive", "HistoricAverage", "AutoARIMA", "Theta", "ZeroModel")
T = TypeVar("T")


@dataclass(slots=True)
class AnchorCase:
    case_id: int
    target_history: list[float]
    target_future: list[float]
    window: list[float]
    best_model: str
    model_errors: dict[str, float]
    model_failures: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AnchorCluster:
    cluster_id: int
    window: list[float]
    best_model: dict[str, int]
    total_weight: int
    medoid_case_id: int
    size: int


@dataclass(slots=True)
class AnchorLibrary:
    version: str
    lookback: int
    horizon: int
    seasonal_period: int
    target_col: str
    timestamp_col: str | None
    dataset_name: str | None
    model_source: str
    cases: list[AnchorCase]
    clusters: list[AnchorCluster]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "seasonal_period": self.seasonal_period,
            "target_col": self.target_col,
            "timestamp_col": self.timestamp_col,
            "dataset_name": self.dataset_name,
            "model_source": self.model_source,
            "cases": [asdict(case) for case in self.cases],
            "clusters": [asdict(cluster) for cluster in self.clusters],
            "case_base": [{"window": case.window, "best_model": case.best_model} for case in self.cases],
            "case_neighbor": [
                {"look_back_window": case.target_history, "pred_window": case.target_future}
                for case in self.cases
            ],
            "cluster_base": [
                {
                    "window": cluster.window,
                    "best_model": cluster.best_model,
                    "total_weight": cluster.total_weight,
                }
                for cluster in self.clusters
            ],
        }

    def save_case_library(self, path: str | Path, data_path: str | Path | None = None) -> Path:
        """Save the library in the old case-library split JSON layout."""
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        write_json(output / "anchor_library.json", payload)
        write_json(output / "case_base.json", payload["case_base"])
        write_json(output / "case_neighbor.json", payload["case_neighbor"])
        write_json(output / "cluster_base.json", payload["cluster_base"])
        write_json(output / "cases_stats.json", dict(Counter(case.best_model for case in self.cases)))
        if data_path is not None:
            write_json(output / "memory.json", build_series_memory_stats(data_path, self.target_col, self.timestamp_col))
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AnchorLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchorLibrary":
        return cls(
            version=str(payload.get("version", "1")),
            lookback=int(payload["lookback"]),
            horizon=int(payload["horizon"]),
            seasonal_period=int(payload["seasonal_period"]),
            target_col=str(payload["target_col"]),
            timestamp_col=payload.get("timestamp_col"),
            dataset_name=payload.get("dataset_name"),
            model_source=str(payload.get("model_source", "foundational_anchorer")),
            cases=[anchor_case_from_dict(case) for case in payload.get("cases", [])],
            clusters=[AnchorCluster(**cluster) for cluster in payload.get("clusters", [])],
        )


def build_anchor_library(
    data_path: str | Path,
    *,
    lookback: int,
    horizon: int,
    seasonal_period: int,
    dataset_name: str | None = None,
    target_col: str | None = None,
    timestamp_col: str | None = None,
    stride: int | None = None,
    max_windows: int | None = None,
    show_progress: bool = False,
    quiet_warnings: bool = False,
) -> AnchorLibrary:
    dataset = load_dataset(data_path, target_col, timestamp_col)
    windows = make_windows(
        dataset,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
    )
    if not windows:
        raise ValueError("no anchor cases could be built; check lookback/horizon against dataset length")

    case_seasonal_period = estimate_periodicity(dataset.frame[dataset.target_col].to_numpy(dtype=float)) if dataset_name else seasonal_period
    cases = [
        build_anchor_case(
            idx,
            window,
            case_seasonal_period,
            dataset_name=dataset_name,
            quiet_warnings=quiet_warnings,
        )
        for idx, window in enumerate(progress_iter(windows, label="Building anchor cases", unit="case", enabled=show_progress))
    ]
    return AnchorLibrary(
        version="1",
        lookback=lookback,
        horizon=horizon,
        seasonal_period=seasonal_period,
        target_col=dataset.target_col,
        timestamp_col=dataset.timestamp_col,
        dataset_name=dataset_name,
        model_source="foundational_anchorer" if dataset_name else "local",
        cases=cases,
        clusters=build_anchor_clusters(cases),
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_series_memory_stats(
    data_path: str | Path,
    target_col: str | None,
    timestamp_col: str | None,
) -> dict[str, Any]:
    dataset = load_dataset(data_path, target_col, timestamp_col)
    values = dataset.frame[dataset.target_col].to_numpy(dtype=float)
    return {
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values)),
        "periodicity_lag": int(estimate_periodicity(values)),
        "series_length": int(len(values)),
        "frequency": infer_frequency_label(dataset.frame, dataset.timestamp_col),
    }


def infer_frequency_label(frame: Any, timestamp_col: str | None) -> str | None:
    if not timestamp_col or timestamp_col not in frame:
        return None
    series = frame[timestamp_col].dropna()
    if len(series) < 2:
        return None
    diffs = series.diff().dropna()
    if diffs.empty:
        return None
    delta = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.iloc[0]
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return None
    if seconds % 86400 == 0:
        days = seconds // 86400
        return "D" if days == 1 else f"{days}D"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return "H" if hours == 1 else f"{hours}H"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes}min"
    return f"{seconds}s"


def progress_iter(items: Sequence[T], *, label: str, unit: str, enabled: bool) -> Iterator[T]:
    if not enabled:
        yield from items
        return

    total = len(items)
    if total == 0:
        return

    try:
        from tqdm.auto import tqdm

        yield from tqdm(items, total=total, desc=label, unit=unit)
        return
    except Exception:
        pass

    if not sys.stderr.isatty():
        yield from items
        return

    width = 32
    for idx, item in enumerate(items, start=1):
        filled = int(width * idx / total)
        bar = "#" * filled + "." * (width - filled)
        sys.stderr.write(f"\r{label}: [{bar}] {idx}/{total} {unit}s")
        sys.stderr.flush()
        yield item
    sys.stderr.write("\n")


def anchor_case_from_dict(payload: dict[str, Any]) -> AnchorCase:
    return AnchorCase(
        case_id=int(payload["case_id"]),
        target_history=list(payload["target_history"]),
        target_future=list(payload["target_future"]),
        window=list(payload.get("window", zscore(np.asarray(payload["target_history"], dtype=float)).tolist())),
        best_model=str(payload["best_model"]),
        model_errors=dict(payload.get("model_errors", {})),
        model_failures=dict(payload.get("model_failures", {})),
    )


def build_anchor_case(
    case_id: int,
    window: TimeSeriesWindow,
    seasonal_period: int,
    dataset_name: str | None = None,
    quiet_warnings: bool = False,
) -> AnchorCase:
    history = window.target_history
    future = window.target_future
    if future is None:
        raise ValueError("anchor cases require future target values")
    with anchorer_warning_context(quiet_warnings):
        forecasts, failures = model_forecasts_for_window(window, len(future), seasonal_period, dataset_name)
    if not forecasts:
        raise ValueError(f"no forecasting models succeeded for anchor case {case_id}; failures={failures}")
    errors = {name: finite_mse(values, future) for name, values in forecasts.items()}
    best_model = min(errors, key=errors.get)
    return AnchorCase(
        case_id=case_id,
        target_history=round_list(history),
        target_future=round_list(future),
        window=round_list(zscore(history)),
        best_model=best_model,
        model_errors={name: round(float(value), 8) for name, value in errors.items()},
        model_failures=failures,
    )


def anchorer_warning_context(enabled: bool):
    if not enabled:
        return null_warning_context()
    return suppress_known_anchorer_warnings()


class null_warning_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class suppress_known_anchorer_warnings:
    def __enter__(self) -> None:
        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r"Too few observations to estimate starting parameters.*",
            category=UserWarning,
            module=r"statsmodels\.tsa\.statespace\.sarimax",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"We recommend keeping prediction length <= 64.*",
            category=UserWarning,
            module=r"chronos\.chronos_bolt",
        )
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._catcher.__exit__(exc_type, exc, traceback)
        return False


def build_anchor_clusters(cases: list[AnchorCase], num_clusters: int = 6) -> list[AnchorCluster]:
    if not cases:
        return []
    k = max(1, min(int(num_clusters), len(cases)))
    vectors = [case.window for case in cases]
    medoid_indices, cluster_indices = kmedoids_cluster(vectors, k)

    clusters: list[AnchorCluster] = []
    for cluster_id, medoid_idx in enumerate(medoid_indices):
        members = [cases[idx] for idx in cluster_indices[cluster_id]]
        counts: dict[str, int] = {}
        for member in members:
            counts[member.best_model] = counts.get(member.best_model, 0) + 1
        filtered = {name: count for name, count in counts.items() if count > 3}
        if not filtered:
            filtered = counts
        total_weight = sum(filtered.values())
        clusters.append(
            AnchorCluster(
                cluster_id=cluster_id,
                window=cases[medoid_idx].window,
                best_model=filtered,
                total_weight=total_weight,
                medoid_case_id=cases[medoid_idx].case_id,
                size=len(members),
            )
        )
    return clusters


def anchor_forecast_from_library(
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    library: AnchorLibrary,
    *,
    top_k: int = 5,
    timestamps: Any = None,
) -> dict[str, Any]:
    if not library.cases:
        raise ValueError("anchor library has no cases")

    cluster = choose_cluster_by_similarity(library.clusters, history)
    current_forecasts, failures = model_forecasts_from_history(
        history,
        horizon,
        seasonal_period,
        dataset_name=library.dataset_name,
        timestamps=timestamps,
        model_names=list(cluster.best_model) if cluster is not None else None,
    )
    weights = {
        name: count / max(cluster.total_weight, 1)
        for name, count in cluster.best_model.items()
        if name in current_forecasts
    } if cluster is not None else {}
    baseline = np.zeros(horizon, dtype=float)
    for name, weight in weights.items():
        if name in current_forecasts:
            baseline += weight * current_forecasts[name]

    return {
        "ensemble_forecast_baseline": round_list(baseline),
        "reference_prediction": round_list(baseline),
        "component_forecasts": {name: round_list(values) for name, values in current_forecasts.items()},
        "weights": {name: round(float(value), 6) for name, value in weights.items()},
        "best_model": cluster.best_model if cluster is not None else {},
        "model_failures": failures,
        "selected_cluster": asdict(cluster) if cluster is not None else None,
        "anchor_cases": [
            {
                "case_id": case.case_id,
                "similarity": round(float(similarity), 6),
                "best_model": case.best_model,
                "model_errors": case.model_errors,
            }
            for case, similarity in ranked_anchor_cases(history, library.cases, top_k=top_k)
        ],
        "anchor_clusters": [asdict(cluster) for cluster in library.clusters],
        "anchor_library": {
            "version": library.version,
            "cases": len(library.cases),
            "lookback": library.lookback,
            "horizon": library.horizon,
            "seasonal_period": library.seasonal_period,
            "dataset_name": library.dataset_name,
            "model_source": library.model_source,
        },
    }


def ranked_anchor_cases(
    history: np.ndarray,
    cases: list[AnchorCase],
    *,
    top_k: int,
) -> list[tuple[AnchorCase, float]]:
    scored = [(case, comprehensive_similarity(history, np.asarray(case.target_history, dtype=float))) for case in cases]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, top_k)]


def model_weights_from_cases(cases: list[tuple[AnchorCase, float]]) -> dict[str, float]:
    raw = dict.fromkeys(MODEL_NAMES, 0.0)
    for case, similarity in cases:
        for name in MODEL_NAMES:
            error = max(float(case.model_errors.get(name, 1e6)), 1e-8)
            raw[name] += max(float(similarity), 0.0) / error
    if not any(value > 0 for value in raw.values()):
        raw = {name: 1.0 for name in MODEL_NAMES}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items() if value > 0}


def default_model_weights(forecasts: dict[str, np.ndarray]) -> dict[str, float]:
    value = 1.0 / len(forecasts)
    return {name: value for name in forecasts}


def model_forecasts_for_window(
    window: TimeSeriesWindow,
    horizon: int,
    seasonal_period: int,
    dataset_name: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    timestamps = window.history[window.timestamp_col] if window.timestamp_col and window.timestamp_col in window.history else None
    return model_forecasts_from_history(
        window.target_history,
        horizon,
        seasonal_period,
        dataset_name=dataset_name,
        timestamps=timestamps,
        strict=True,
    )


def model_forecasts_from_history(
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    dataset_name: str | None = None,
    timestamps: Any = None,
    model_names: list[str] | None = None,
    strict: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if dataset_name:
        result = forecast_with_anchorer_models(
            dataset_name,
            history,
            horizon,
            seasonal_period,
            timestamps=timestamps,
            model_names=model_names,
            strict=strict,
        )
        return result.forecasts, result.failures
    return primitive_model_forecasts(history, horizon, seasonal_period), {}


def primitive_model_forecasts(history: np.ndarray, horizon: int, seasonal_period: int) -> dict[str, np.ndarray]:
    y = np.asarray(history, dtype=float)
    seasonal = max(1, seasonal_period)
    forecasts: dict[str, np.ndarray] = {
        "SeasonalNaive": seasonal_naive_forecast(y, horizon, seasonal),
        "HistoricAverage": np.repeat(np.mean(y), horizon).astype(float),
        "ZeroModel": np.zeros(horizon, dtype=float),
    }
    for name, fn in {
        "AutoARIMA": auto_arima_forecast,
        "Theta": theta_forecast,
    }.items():
        try:
            forecasts[name] = fn(y, horizon, seasonal)
        except Exception:
            continue
    return forecasts


def kmedoids_cluster(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[int]]]:
    try:
        from pyclustering.cluster.kmedoids import kmedoids

        import random

        random.seed(0)
        initial = random.sample(range(len(vectors)), k)
        instance = kmedoids(vectors, initial, ccore=False)
        instance.process()
        return [int(idx) for idx in instance.get_medoids()], [[int(i) for i in group] for group in instance.get_clusters()]
    except Exception:
        medoids = evenly_spaced_indices(len(vectors), k)
        clusters = [[] for _ in medoids]
        matrix = np.asarray(vectors, dtype=float)
        for idx, vector in enumerate(matrix):
            distances = [float(np.linalg.norm(vector - matrix[medoid])) for medoid in medoids]
            clusters[int(np.argmin(distances))].append(idx)
        return medoids, clusters


def evenly_spaced_indices(size: int, k: int) -> list[int]:
    if k <= 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, size - 1, k)]


def choose_cluster_by_similarity(clusters: list[AnchorCluster], history: np.ndarray) -> AnchorCluster | None:
    if not clusters:
        return None
    query = zscore(np.asarray(history, dtype=float))
    return max(clusters, key=lambda cluster: comprehensive_similarity(query, np.asarray(cluster.window, dtype=float)))


def finite_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    value = mse(prediction, target)
    return float(value) if np.isfinite(value) else 1e12


def seasonal_naive_forecast(y: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    tail = y[-seasonal_period:]
    repeats = int(np.ceil(horizon / len(tail)))
    return np.tile(tail, repeats)[:horizon].astype(float)


def linear_drift_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    slope = linear_slope(y[-min(len(y), 24) :])
    steps = np.arange(1, horizon + 1, dtype=float)
    return y[-1] + slope * steps


def auto_arima_forecast(y: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    seasonal_order = (1, 1, 1, int(seasonal_period)) if seasonal_period > 1 and len(y) > seasonal_period * 2 else (0, 0, 0, 0)
    fitted = ARIMA(y, order=(1, 1, 1), seasonal_order=seasonal_order).fit()
    return np.asarray(fitted.forecast(steps=horizon), dtype=float)


def theta_forecast(y: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel

    period = int(seasonal_period) if seasonal_period > 1 and len(y) >= seasonal_period * 2 else None
    fitted = ThetaModel(y, period=period).fit()
    return np.asarray(fitted.forecast(horizon), dtype=float)


def comprehensive_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = align_length(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    right = align_length(np.asarray(right, dtype=float), left)
    if len(left) < 2 or len(right) < 2:
        return 0.0
    cosine = cosine_similarity(zscore(left), zscore(right))
    trend = trend_similarity(left, right)
    volatility = volatility_similarity(left, right)
    shape = pattern_similarity(left, right)
    return float(np.clip(0.40 * cosine + 0.25 * trend + 0.20 * volatility + 0.15 * shape, 0.0, 1.0))


def align_length(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(values) == len(reference):
        return values
    if len(values) > len(reference):
        return values[-len(reference) :]
    pad = np.repeat(values[0], len(reference) - len(values))
    return np.concatenate([pad, values])


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def estimate_periodicity(values: np.ndarray, max_lag: int | None = None) -> int:
    y = np.asarray(values, dtype=float)
    if len(y) < 3:
        return 1
    if max_lag is None:
        max_lag = max(2, len(y) // 2)
    y = y - float(np.mean(y))
    autocorr = np.correlate(y, y, mode="full")[len(y) - 1 : len(y) - 1 + max_lag]
    if len(autocorr) < 2:
        return 1
    lag = int(np.argmax(autocorr[1:]) + 1)
    return max(1, lag)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return 0.5 + 0.5 * float(np.dot(left, right) / denom)


def trend_similarity(left: np.ndarray, right: np.ndarray) -> float:
    spread = float(max(np.ptp(left), np.ptp(right), 1e-8))
    diff = abs(linear_slope(left) - linear_slope(right))
    return float(np.exp(-diff / spread))


def volatility_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vol = float(np.std(np.diff(left))) if len(left) > 1 else 0.0
    right_vol = float(np.std(np.diff(right))) if len(right) > 1 else 0.0
    denom = max(left_vol, right_vol, 1e-8)
    return float(np.exp(-abs(left_vol - right_vol) / denom))


def pattern_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_diff = np.sign(np.diff(left))
    right_diff = np.sign(np.diff(right))
    if len(left_diff) == 0:
        return 0.0
    return float(np.mean(left_diff == right_diff))


def linear_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y.astype(float), 1)[0])


def round_list(values: list[float] | np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]
