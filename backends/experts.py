"""Configurable forecasting experts for offline anchor-library construction."""

from __future__ import annotations

import importlib
import json
import re
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


BUILTIN_EXPERTS = (
    {"name": "SeasonalNaive", "kind": "seasonal_naive"},
    {"name": "HistoricAverage", "kind": "historic_average"},
    {"name": "AutoARIMA", "kind": "auto_arima"},
    {"name": "Theta", "kind": "theta"},
    {"name": "ZeroModel", "kind": "zero"},
)

_KINDS = {
    "seasonal_naive",
    "historic_average",
    "zero",
    "auto_arima",
    "theta",
    "ridge",
    "xgboost",
    "lightgbm",
    "chronos",
    "chronos_bolt",
    "dlinear",
    "python",
}
_PARAMETERS = {
    "seasonal_naive": {"season_length"},
    "historic_average": {"window"},
    "zero": set(),
    "auto_arima": {"max_p", "max_q", "d_values"},
    "theta": {"deseasonalize", "period"},
    "ridge": {"lags", "alpha"},
    "xgboost": {
        "lags",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "reg_lambda",
        "seed",
    },
    "lightgbm": {
        "lags",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "reg_lambda",
        "seed",
    },
    "chronos": {"num_samples", "temperature", "top_k", "top_p", "seed"},
    "chronos_bolt": set(),
    "dlinear": set(),
}
_CHRONOS_LOCK = RLock()


@dataclass
class _ArimaOrderCache:
    maximum: int
    records: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    fits: int = 0


_ARIMA_ORDER_CACHE: ContextVar[_ArimaOrderCache | None] = ContextVar(
    "arima_order_cache", default=None
)


@contextmanager
def arima_order_cache(max_entries: int = 8192):
    """Reuse identical order fits only within an explicit offline preparation scope."""
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or not 1 <= max_entries <= 65536
    ):
        raise ValueError("ARIMA cache size must be an integer between one and 65536")
    active = _ARIMA_ORDER_CACHE.get()
    if active is not None:
        yield active
        return
    cache = _ArimaOrderCache(max_entries)
    token = _ARIMA_ORDER_CACHE.set(cache)
    try:
        yield cache
    finally:
        _ARIMA_ORDER_CACHE.reset(token)
        cache.records.clear()


def _remember_arima(cache: _ArimaOrderCache | None, key: tuple, value: Any) -> None:
    if cache is None:
        return
    cache.records[key] = value
    cache.records.move_to_end(key)
    while len(cache.records) > cache.maximum:
        cache.records.popitem(last=False)


def load_expert_manifest(path: str | Path | None) -> list[dict[str, Any]]:
    """Read expert definitions; weight files and model libraries remain external."""
    if path is None:
        return [dict(spec) for spec in BUILTIN_EXPERTS]
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    specs = payload.get("experts") if isinstance(payload, dict) else payload
    return validate_expert_specs(specs)


def validate_expert_specs(specs: object) -> list[dict[str, Any]]:
    if not isinstance(specs, list) or not specs:
        raise ValueError("expert manifest must contain a nonempty 'experts' list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in specs:
        if not isinstance(raw, dict):
            raise ValueError("each forecasting expert must be an object")
        spec = dict(raw)
        name, kind = spec.get("name"), spec.get("kind")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("forecasting expert names must be nonempty and unique")
        if kind not in _KINDS:
            raise ValueError(f"unsupported expert kind for {name!r}: {kind!r}")
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"expert {name!r} params must be an object")
        if kind != "python":
            unknown = set(params) - _PARAMETERS[kind]
            if unknown:
                raise ValueError(f"unknown parameters for {name!r}: {sorted(unknown)}")
            _validate_parameters(kind, params)
        if kind in {"chronos", "chronos_bolt", "dlinear"} and (
            not isinstance(spec.get("model_path"), str)
            or not spec["model_path"].strip()
        ):
            raise ValueError(f"{kind} requires an explicit prepared local model_path")
        if kind in {"chronos", "chronos_bolt", "dlinear"}:
            spec["device"] = validate_device(spec.get("device", "cpu"))
        if kind in {"chronos_bolt", "dlinear"} and not re.fullmatch(
            r"[0-9a-f]{64}", str(spec.get("model_identity", ""))
        ):
            raise ValueError(f"{kind} requires its prepared model identity")
        if kind == "python":
            entrypoint = spec.get("entrypoint")
            if not isinstance(entrypoint, str) or len(entrypoint.split(":")) != 2:
                raise ValueError(
                    f"Python expert {name!r} requires entrypoint='module:callable'"
                )
            if not isinstance(spec.get("kwargs", {}), dict):
                raise ValueError(f"Python expert {name!r} kwargs must be an object")
        seen.add(name)
        result.append(spec)
    return result


def validate_device(device: str) -> str:
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError("expert device must be cpu, cuda, or cuda:<index>")
    return "cuda:0" if device == "cuda" else device


def forecast_expert_specs(
    specs: list[dict[str, Any]],
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    *,
    timestamps: Any = None,
    model_names: list[str] | None = None,
    strict: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Run named experts and reject partial, nonfinite, or incorrectly sized outputs."""
    history = finite_vector(history, "expert history")
    if horizon < 1 or seasonal_period < 1:
        raise ValueError("expert horizon and seasonal_period must be positive")
    definitions = {spec["name"]: spec for spec in validate_expert_specs(specs)}
    selected = (
        list(definitions) if model_names is None else list(dict.fromkeys(model_names))
    )
    if not selected or any(name not in definitions for name in selected):
        raise ValueError("requested expert pool is empty or absent from its manifest")
    predictions: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for name in selected:
        try:
            values = forecast_expert(
                definitions[name],
                history,
                horizon,
                seasonal_period,
                timestamps=timestamps,
            )
            prediction = finite_vector(values, f"expert {name!r} forecast")
            if len(prediction) != horizon:
                raise ValueError(
                    f"expected {horizon} forecast values, got {len(prediction)}"
                )
            predictions[name] = prediction
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"forecasting expert {name!r} failed; verify its dependencies and weights"
                ) from exc
            failures[name] = f"{type(exc).__name__}: {exc}"
    return predictions, failures


def forecast_expert(
    spec: dict[str, Any],
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    *,
    timestamps: Any = None,
) -> np.ndarray:
    history = finite_vector(history, "expert history")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("forecast horizon must be a positive integer")
    if (
        isinstance(seasonal_period, bool)
        or not isinstance(seasonal_period, int)
        or seasonal_period <= 0
    ):
        raise ValueError("seasonal_period must be a positive integer")
    spec = validate_expert_specs([spec])[0]
    kind, params = spec["kind"], spec.get("params", {})
    if kind == "seasonal_naive":
        period = params.get("season_length", seasonal_period)
        tail = history[-min(period, len(history)) :]
        return np.resize(tail, horizon).astype(float)
    if kind == "historic_average":
        size = params.get("window", len(history))
        return np.full(horizon, np.mean(history[-size:]), dtype=float)
    if kind == "zero":
        return np.zeros(horizon, dtype=float)
    if kind == "auto_arima":
        return auto_arima_forecast(history, horizon, seasonal_period, **params)
    if kind == "theta":
        from statsmodels.tsa.forecasting.theta import ThetaModel

        requested = params.get("period", seasonal_period)
        period = requested if requested > 1 and len(history) >= 2 * requested else None
        deseasonalize = params.get("deseasonalize", True) and period is not None
        if np.ptp(history) == 0:
            return np.full(horizon, history[-1], dtype=float)
        fitted = ThetaModel(history, period=period, deseasonalize=deseasonalize).fit()
        return np.asarray(fitted.forecast(horizon), dtype=float)
    if kind in {"ridge", "xgboost", "lightgbm"}:
        return _autoregressive_forecast(kind, history, horizon, params)
    if kind in {"chronos", "chronos_bolt"}:
        return _chronos_forecast(spec, history, horizon)
    if kind == "dlinear":
        from backends.neural import dlinear_forecast

        return dlinear_forecast(spec, history, horizon, timestamps=timestamps)
    if kind == "python":
        # The manifest is a trusted runtime configuration, not an uploaded data file.
        module_name, attribute = spec["entrypoint"].split(":", 1)
        predictor = getattr(importlib.import_module(module_name), attribute)
        if not callable(predictor):
            raise TypeError(f"expert entrypoint is not callable: {spec['entrypoint']}")
        return finite_vector(
            predictor(
                history=history.copy(),
                horizon=horizon,
                seasonal_period=seasonal_period,
                timestamps=(
                    timestamps.copy() if hasattr(timestamps, "copy") else timestamps
                ),
                **spec.get("kwargs", {}),
            ),
            f"Python expert {spec['name']!r}",
        )
    raise ValueError(f"unsupported expert kind: {kind!r}")


def auto_arima_forecast(
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    *,
    max_p: int = 2,
    max_q: int = 2,
    d_values: list[int] | tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """Select a small nonseasonal ARIMA candidate set by AIC on the lookback."""
    from statsmodels.tsa.arima.model import ARIMA

    history = np.ascontiguousarray(
        finite_vector(history, "ARIMA history"), dtype=np.float64
    )
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("ARIMA horizon must be a positive integer")
    if np.ptp(history) == 0:
        return np.full(horizon, history[-1], dtype=float)
    best = None
    best_aic = float("inf")
    cache = _ARIMA_ORDER_CACHE.get()
    context = (len(history), history.tobytes(), horizon)
    for p in range(max_p + 1):
        for d in d_values:
            for q in range(max_q + 1):
                if len(history) <= p + d + q + 3:
                    continue
                key = (*context, p, d, q)
                try:
                    if cache is not None and key in cache.records:
                        cache.hits += 1
                        cached = cache.records[key]
                        cache.records.move_to_end(key)
                        if cached is None:
                            continue
                        score, pred = cached
                        if score >= best_aic:
                            continue
                        if pred is not None:
                            best, best_aic = pred.copy(), score
                            continue
                    if cache is not None:
                        cache.fits += 1
                    candidate = ARIMA(history, order=(p, d, q)).fit()
                    if not candidate.mle_retvals.get("converged", True):
                        _remember_arima(cache, key, None)
                        continue
                    score = float(candidate.aic)
                    if not np.isfinite(score):
                        _remember_arima(cache, key, None)
                        continue
                    if score < best_aic:
                        pred = np.asarray(candidate.forecast(horizon), dtype=float)
                        if pred.shape == (horizon,) and np.isfinite(pred).all():
                            _remember_arima(cache, key, (score, pred.copy()))
                            best, best_aic = pred.copy(), score
                        else:
                            _remember_arima(cache, key, None)
                    else:
                        _remember_arima(cache, key, (score, None))
                except (
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    np.linalg.LinAlgError,
                ):
                    _remember_arima(cache, key, None)
                    continue
    if best is None:
        raise ValueError("no finite ARIMA candidate could be fitted to the lookback")
    return best


def _validate_parameters(kind: str, params: dict[str, Any]) -> None:
    positive_ints = {
        "season_length",
        "window",
        "period",
        "lags",
        "n_estimators",
        "max_depth",
        "num_samples",
    }
    nonnegative_ints = {"max_p", "max_q", "top_k", "seed"}
    for key, value in params.items():
        if key in positive_ints | nonnegative_ints:
            lower = 1 if key in positive_ints else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{kind}.{key} must be an integer >= {lower}")
        elif key in {"alpha", "learning_rate", "reg_lambda", "temperature", "top_p"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{kind}.{key} must be finite")
            if value < 0 or (
                key in {"learning_rate", "temperature", "top_p"} and value == 0
            ):
                raise ValueError(f"{kind}.{key} is outside its valid range")
            if key == "top_p" and value > 1:
                raise ValueError("chronos.top_p must not exceed one")
        elif key == "deseasonalize" and not isinstance(value, bool):
            raise ValueError("theta.deseasonalize must be boolean")
        elif key == "d_values":
            if (
                not isinstance(value, (tuple, list))
                or not value
                or any(
                    isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2)
                    for v in value
                )
            ):
                raise ValueError(
                    "auto_arima.d_values must be a nonempty sequence from {0,1,2}"
                )
    if params.get("max_p", 0) > 5 or params.get("max_q", 0) > 5:
        raise ValueError("ARIMA search bounds must not exceed five")
    if params.get("n_estimators", 1) > 2000 or params.get("num_samples", 1) > 1000:
        raise ValueError("expert search exceeds bounded inference limits")


def _autoregressive_forecast(
    kind: str, history: np.ndarray, horizon: int, params: dict[str, Any]
) -> np.ndarray:
    """Fit solely on the supplied lookback and forecast recursively."""
    lags = params.get("lags", min(12, max(1, len(history) // 4)))
    if lags >= len(history) - 1:
        raise ValueError("autoregressive lags must leave at least two training rows")
    center = float(np.mean(history))
    scale = float(np.std(history))
    if scale == 0:
        return np.full(horizon, history[-1], dtype=float)
    normalized = (history - center) / scale
    x = np.asarray([normalized[i - lags : i] for i in range(lags, len(normalized))])
    y = normalized[lags:]
    if kind == "ridge":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(lags + 1) * params.get("alpha", 1.0)
        penalty[0, 0] = 0
        coefficients = np.linalg.lstsq(
            design.T @ design + penalty, design.T @ y, rcond=None
        )[0]
        predict = lambda row: float(np.r_[1.0, row] @ coefficients)
    else:
        kwargs = dict(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=params.get("seed", 0),
            n_jobs=1,
        )
        if kind == "xgboost":
            from xgboost import XGBRegressor

            fitted = XGBRegressor(objective="reg:squarederror", verbosity=0, **kwargs)
        else:
            from lightgbm import LGBMRegressor

            fitted = LGBMRegressor(
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
                min_child_samples=min(20, max(2, len(x) // 5)),
                **kwargs,
            )
        fitted.fit(x, y)
        predict = lambda row: float(fitted.predict(np.asarray(row).reshape(1, -1))[0])
    values = list(normalized)
    for _ in range(horizon):
        value = predict(values[-lags:])
        if not np.isfinite(value):
            raise ValueError("autoregressive prediction became nonfinite")
        values.append(value)
    return np.asarray(values[-horizon:]) * scale + center


@lru_cache(maxsize=2)
def _load_chronos(
    model_path: str, identity: str, kind: str = "chronos", device: str = "cpu"
):
    import torch
    from chronos import BaseChronosPipeline, ChronosPipeline
    from backends.catalog import chronos_assets, content_fingerprint

    if device.startswith("cuda") and (
        not torch.cuda.is_available()
        or int(device.split(":")[1]) >= torch.cuda.device_count()
    ):
        raise ValueError("requested Chronos CUDA device is unavailable")
    if (
        identity
        and content_fingerprint(chronos_assets(Path(model_path), expected_kind=kind))
        != identity
    ):
        raise ValueError(
            "Chronos weight files differ from the prepared expert identity"
        )
    pipeline = BaseChronosPipeline if kind == "chronos_bolt" else ChronosPipeline
    loaded = pipeline.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float32, local_files_only=True
    )
    model = loaded.model.model if kind == "chronos" else loaded.model
    _validate_loaded_chronos_weights(Path(model_path), model)
    return loaded


def _validate_loaded_chronos_weights(path: Path, model: Any) -> None:
    """Reject incomplete checkpoints instead of accepting randomly initialized keys."""
    shapes: dict[str, tuple[int, ...]] = {}
    safe_files = sorted(path.glob("*.safetensors"))
    if safe_files:
        from safetensors import safe_open

        for file in safe_files:
            with safe_open(file, framework="pt", device="cpu") as state:
                for key in state.keys():
                    if key in shapes:
                        raise ValueError(
                            "Chronos checkpoint contains duplicate parameter keys"
                        )
                    shapes[key] = tuple(state.get_slice(key).get_shape())
    else:
        import torch

        for file in sorted(path.glob("pytorch_model*.bin")):
            state = torch.load(file, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Chronos weight file must contain a parameter mapping")
            for key, tensor in state.items():
                if key in shapes or not hasattr(tensor, "shape"):
                    raise ValueError(
                        "Chronos checkpoint has malformed or duplicate parameters"
                    )
                shapes[key] = tuple(tensor.shape)
    expected = model.state_dict()
    if set(shapes).difference(expected):
        raise ValueError("Chronos checkpoint contains unexpected architecture keys")
    for name, parameter in model.named_parameters():
        if name not in shapes or shapes[name] != tuple(parameter.shape):
            raise ValueError(
                f"Chronos checkpoint is missing a complete matching parameter: {name}"
            )


def _chronos_forecast(
    spec: dict[str, Any], history: np.ndarray, horizon: int
) -> np.ndarray:
    import torch

    path = Path(spec["model_path"]).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Chronos weights are not prepared at {path}; run initialize first"
        )
    params = spec.get("params", {})
    device = validate_device(spec.get("device", "cpu"))
    kind = spec["kind"]
    # Sampling state and the CPU pipeline are shared only under this lock.
    cuda_devices = [int(device.split(":")[1])] if device.startswith("cuda") else []
    with _CHRONOS_LOCK, torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(params.get("seed", 0))
        if cuda_devices:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(
                params.get("seed", 0)
            )
        pipeline = _load_chronos(
            str(path), str(spec.get("model_identity", "")), kind, device
        )
        if kind == "chronos_bolt":
            quantiles, _ = pipeline.predict_quantiles(
                torch.as_tensor(history, dtype=torch.float32),
                prediction_length=horizon,
                quantile_levels=[0.5],
                limit_prediction_length=False,
            )
            values = quantiles.detach().cpu().numpy()
            if values.shape != (1, horizon, 1) or not np.isfinite(values).all():
                raise ValueError(
                    "Chronos-Bolt returned an invalid median forecast tensor"
                )
            return values[0, :, 0].astype(float)
        samples = pipeline.predict(
            torch.as_tensor(history, dtype=torch.float32),
            prediction_length=horizon,
            num_samples=params.get("num_samples", 20),
            temperature=params.get("temperature", 1.0),
            top_k=params.get("top_k", 50),
            top_p=params.get("top_p", 1.0),
        )
        values = samples.detach().cpu().numpy()
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or values.shape[2] != horizon
        or not np.isfinite(values).all()
    ):
        raise ValueError("Chronos returned an invalid sample forecast tensor")
    return np.median(values[0], axis=0).astype(float)


def finite_vector(values: Any, label: str) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) for v in raw.flat):
        raise ValueError(f"{label} cannot contain boolean values")
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty one-dimensional finite sequence")
    return result
