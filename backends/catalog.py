"""Named local expert configurations and explicitly requested model preparation."""

from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
import math
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from backends.experts import validate_expert_specs
from utils.io import atomic_write_json


# Candidate models may be excluded according to task requirements.
DEFAULT_MODELS = ("SeasonalNaive", "HistoricAverage", "AutoARIMA", "Theta")
BALANCED_MODELS = (
    "SeasonalNaive",
    "HistoricAverage",
    "AutoARIMA",
    "Theta",
    "Ridge",
    "DLinear",
    "ChronosBoltBase",
)
_CATALOG = {
    "SeasonalNaive": {
        "kind": "seasonal_naive",
        "dependencies": [],
        "fit_scope": "lookback",
    },
    "HistoricAverage": {
        "kind": "historic_average",
        "dependencies": [],
        "fit_scope": "lookback",
    },
    "AutoARIMA": {
        "kind": "auto_arima",
        "dependencies": ["statsmodels"],
        "fit_scope": "lookback",
    },
    "Theta": {
        "kind": "theta",
        "dependencies": ["statsmodels"],
        "fit_scope": "lookback",
    },
    "ZeroModel": {"kind": "zero", "dependencies": [], "fit_scope": "constant"},
    "Ridge": {"kind": "ridge", "dependencies": [], "fit_scope": "lookback"},
    "XGBoost": {
        "kind": "xgboost",
        "dependencies": ["xgboost"],
        "fit_scope": "lookback",
    },
    "LightGBM": {
        "kind": "lightgbm",
        "dependencies": ["lightgbm"],
        "fit_scope": "lookback",
    },
    "ChronosTiny": {
        "kind": "chronos",
        "dependencies": ["torch", "chronos", "transformers", "huggingface_hub"],
        "fit_scope": "pretrained_frozen",
        "repository": "amazon/chronos-t5-tiny",
    },
    "ChronosBoltBase": {
        "kind": "chronos_bolt",
        "dependencies": [
            "torch",
            "chronos",
            "transformers",
            "huggingface_hub",
            "safetensors",
        ],
        "fit_scope": "pretrained_frozen",
        "repository": "amazon/chronos-bolt-base",
    },
    "DLinear": {
        "kind": "dlinear",
        "dependencies": ["torch", "safetensors"],
        "fit_scope": "training_prefix",
    },
}
_PACKAGES = {"chronos": "chronos-forecasting", "huggingface_hub": "huggingface-hub"}


def list_model_catalog() -> list[dict[str, Any]]:
    """Return capabilities without preparing assets, importing models, or using network."""
    return [
        dict(
            name=name,
            **copy.deepcopy(entry),
            default=name in DEFAULT_MODELS,
            balanced_preset=name in BALANCED_MODELS,
            local_weights_required=entry["kind"] in {"chronos", "chronos_bolt"},
        )
        for name, entry in _CATALOG.items()
    ]


def selected_model_names(models: Sequence[str] | None) -> list[str]:
    if isinstance(models, str):
        raise ValueError(
            "models must be a sequence of names, not a comma-separated string"
        )
    aliases = {name.casefold(): name for name in _CATALOG}
    names = list(DEFAULT_MODELS) if models is None else list(models)
    if not names:
        raise ValueError("select at least one expert model")
    result = []
    for name in names:
        if not isinstance(name, str) or name.strip().casefold() not in aliases:
            raise ValueError(
                f"unknown expert {name!r}; choose from {', '.join(_CATALOG)}"
            )
        canonical = aliases[name.strip().casefold()]
        if canonical in result:
            raise ValueError(f"duplicate expert selection: {canonical}")
        result.append(canonical)
    return result


def preflight_models(
    names: Sequence[str],
    model_store: str | Path,
    *,
    download_models: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Check all requested dependencies and local assets before any model is loaded."""
    store = Path(model_store).expanduser().resolve()
    names = selected_model_names(names)
    from backends.experts import validate_device

    device = validate_device(device)
    if device.startswith("cuda") and any(
        _CATALOG[name]["kind"] in {"chronos", "chronos_bolt", "dlinear"}
        for name in names
    ):
        try:
            import torch
        except ImportError as exc:
            raise ValueError("CUDA forecasting experts require PyTorch") from exc
        if (
            not torch.cuda.is_available()
            or int(device.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise ValueError("requested forecasting expert CUDA device is unavailable")
    if store.exists() and not store.is_dir():
        raise ValueError(f"model_store is not a directory: {store}")
    details = {}
    errors = []
    for name in names:
        entry = _CATALOG[name]
        folder = store / name
        if folder.exists() and not folder.is_dir():
            errors.append(f"{name}: model path is not a directory: {folder}")
        packages = {}
        for module in entry["dependencies"]:
            try:
                available = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                available = False
            distribution = _PACKAGES.get(module, module)
            if not available:
                errors.append(f"{name} requires {distribution}")
            try:
                packages[distribution] = version(distribution)
            except PackageNotFoundError:
                packages[distribution] = "unversioned" if available else "missing"
        assets = {}
        if entry["kind"] in {"chronos", "chronos_bolt"}:
            try:
                assets = chronos_assets(folder, expected_kind=entry["kind"])
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                if not download_models or isinstance(exc, ModelAssetMismatchError):
                    errors.append(
                        f"{name}: {exc}. Prepare weights at {folder} or explicitly enable --download-models"
                    )
        details[name] = {"dependencies": packages, "existing_assets": assets}
    if errors:
        raise ValueError("model preflight failed:\n- " + "\n- ".join(errors))
    return details


def prepare_models(
    names: Sequence[str],
    model_store: str | Path,
    *,
    download_models: bool = False,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Persist model recipes; download pretrained weights only with explicit permission."""
    names = selected_model_names(names)
    from backends.experts import validate_device

    device = validate_device(device)
    if "DLinear" in names:
        raise ValueError(
            "DLinear requires dataset-specific prepare_dlinear_experts after source-prefix preflight"
        )
    details = preflight_models(
        names, model_store, download_models=download_models, device=device
    )
    store = Path(model_store).expanduser().resolve()
    result = []
    for name in names:
        entry = _CATALOG[name]
        folder = store / name
        folder.mkdir(parents=True, exist_ok=True)
        spec: dict[str, Any] = {"name": name, "kind": entry["kind"]}
        assets = details[name]["existing_assets"]
        if entry["kind"] in {"chronos", "chronos_bolt"}:
            if not assets:
                if not download_models:
                    raise FileNotFoundError(
                        f"pretrained weights not available at {folder}"
                    )
                if folder.is_symlink():
                    raise ValueError(
                        "cannot download through an external model symlink; prepare its complete weights first"
                    )
                from huggingface_hub import snapshot_download

                try:
                    snapshot_download(
                        repo_id=entry["repository"],
                        local_dir=str(folder),
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "pytorch_model*.bin",
                            "README.md",
                            "LICENSE*",
                        ],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not prepare {name} from {entry['repository']}. "
                        f"Check repository access or place its complete weights at {folder}, then resume."
                    ) from exc
                assets = chronos_assets(folder, expected_kind=entry["kind"])
            spec["model_path"] = str(folder)
            spec["model_identity"] = content_fingerprint(assets)
            spec["device"] = device
            spec["family"] = name
            spec["params"] = (
                {}
                if entry["kind"] == "chronos_bolt"
                else {
                    "num_samples": 20,
                    "temperature": 1.0,
                    "top_k": 50,
                    "top_p": 1.0,
                    "seed": 0,
                }
            )
        record = {
            "name": name,
            "fit_scope": entry["fit_scope"],
            "expert": spec,
            "dependencies": details[name]["dependencies"],
            "assets": assets,
        }
        if "repository" in entry:
            record["repository"] = entry["repository"]
        record_path = (
            store / f"{name}.prepared_model.json"
            if folder.is_symlink()
            else folder / "prepared_model.json"
        )
        atomic_write_json(record_path, record)
        result.append(spec)
    return validate_expert_specs(result)


def candidate_parameters(
    spec: dict[str, Any],
    *,
    lookback: int,
    seasonal_period: int,
    tuning_rounds: int,
    seed: int,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    """Return distinct recipes with an unchanged baseline and stable budget prefixes.

    The budget counts attempted recipes. Finite simple-model spaces stop when
    exhausted. Stochastic model seeds are held fixed across all candidates.
    """
    if (
        isinstance(tuning_rounds, bool)
        or not isinstance(tuning_rounds, int)
        or not 1 <= tuning_rounds <= 256
    ):
        raise ValueError("tuning_rounds must be an integer between 1 and 256")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if horizon is not None and (
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer or None")
    space = _candidate_space(spec, lookback, seasonal_period, seed)
    import numpy as np

    result = [dict(space.baseline)]
    budget = min(tuning_rounds, space.capacity)
    if budget == 1:
        return result
    seen = {content_fingerprint(space.effective_baseline or space.baseline)}
    kind = spec["kind"]
    if kind in {"seasonal_naive", "historic_average"}:
        key = "season_length" if kind == "seasonal_naive" else "window"
        for period in (
            seasonal_period,
            lookback,
            horizon if horizon is not None else spec.get("horizon", lookback),
            seasonal_period // 2,
            seasonal_period * 2,
            seasonal_period * 3,
            seasonal_period * 7,
            1,
        ):
            if not isinstance(period, int) or not 1 <= period <= lookback:
                continue
            params = {key: period}
            fingerprint = content_fingerprint(params)
            if fingerprint not in seen:
                result.append(params)
                seen.add(fingerprint)
            if len(result) >= budget:
                return result
    rng = np.random.default_rng(seed)
    for index in rng.permutation(space.alternative_count):
        params = space.alternative(int(index))
        fingerprint = content_fingerprint(params)
        if space.is_baseline(params) or fingerprint in seen:
            continue
        result.append(params)
        seen.add(fingerprint)
        if len(result) >= budget:
            break
    return result


def candidate_capacity(
    spec: dict[str, Any],
    *,
    lookback: int,
    seasonal_period: int,
    horizon: int | None = None,
) -> int:
    """Return the actual finite search-space size before applying a tuning budget."""
    if horizon is not None and (
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer or None")
    return _candidate_space(spec, lookback, seasonal_period).capacity


@dataclass(slots=True)
class _CandidateSpace:
    baseline: dict[str, Any]
    explicit_alternatives: list[dict[str, Any]] | None = None
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    fixed: dict[str, Any] = field(default_factory=dict)
    effective_baseline: dict[str, Any] | None = None

    @property
    def alternative_count(self) -> int:
        if self.explicit_alternatives is not None:
            return len(self.explicit_alternatives)
        return math.prod(len(values) for _, values in self.axes)

    @property
    def capacity(self) -> int:
        if self.explicit_alternatives is not None:
            return 1 + len(self.explicit_alternatives)
        baseline = self.effective_baseline or self.baseline
        included = (
            set(baseline) == set(self.fixed) | {name for name, _ in self.axes}
            and all(baseline.get(name) == value for name, value in self.fixed.items())
            and all(baseline.get(name) in values for name, values in self.axes)
        )
        return self.alternative_count + (0 if included else 1)

    def is_baseline(self, params: dict[str, Any]) -> bool:
        return params == (self.effective_baseline or self.baseline)

    def alternative(self, index: int) -> dict[str, Any]:
        if self.explicit_alternatives is not None:
            return dict(self.explicit_alternatives[index])
        params = dict(self.fixed)
        for name, values in reversed(self.axes):
            index, digit = divmod(index, len(values))
            params[name] = values[digit]
        return params


def _candidate_space(
    spec: dict[str, Any], lookback: int, seasonal_period: int, seed: int = 0
) -> _CandidateSpace:
    for name, value, minimum in (
        ("lookback", lookback, 2),
        ("seasonal_period", seasonal_period, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    kind = spec["kind"]
    if kind == "seasonal_naive":
        baseline = {"season_length": seasonal_period}
        return _CandidateSpace(
            baseline,
            [
                {"season_length": period}
                for period in range(1, lookback + 1)
                if period != min(seasonal_period, lookback)
            ],
            effective_baseline={"season_length": min(seasonal_period, lookback)},
        )
    if kind == "historic_average":
        return _CandidateSpace(
            {"window": lookback}, axes=(("window", tuple(range(1, lookback + 1))),)
        )
    if kind == "zero":
        return _CandidateSpace({}, [])
    if kind in {"dlinear", "chronos_bolt"}:
        return _CandidateSpace(dict(spec.get("params", {})), [])
    if kind == "theta":
        baseline = {"deseasonalize": True, "period": seasonal_period}
        baseline_period = (
            seasonal_period if 1 < seasonal_period <= lookback // 2 else None
        )
        alternatives = []
        if baseline_period is not None:
            alternatives.append({"deseasonalize": False, "period": seasonal_period})
        alternatives.extend(
            {"deseasonalize": True, "period": period}
            for period in range(2, lookback // 2 + 1)
            if period != baseline_period
        )
        return _CandidateSpace(baseline, alternatives)
    if kind == "auto_arima":
        baseline = {"max_p": 2, "max_q": 2, "d_values": [0, 1]}
        if lookback < 4:
            raise ValueError("AutoARIMA requires lookback >= 4")
        difference_sets = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))

        def orders(params):
            return tuple(
                (p, d, q)
                for p in range(params["max_p"] + 1)
                for d in params["d_values"]
                for q in range(params["max_q"] + 1)
                if lookback > p + d + q + 3
            )

        seen = {orders(baseline)}
        alternatives = []
        for p, q, differences in product(range(6), range(6), difference_sets):
            params = {"max_p": p, "max_q": q, "d_values": list(differences)}
            feasible = orders(params)
            if feasible and feasible not in seen:
                seen.add(feasible)
                alternatives.append(params)
        return _CandidateSpace(baseline, alternatives)
    if kind in {"ridge", "xgboost", "lightgbm"}:
        if lookback < 3:
            raise ValueError("autoregressive expert tuning requires lookback >= 3")
        short = min(12, max(1, lookback // 4))
        cap = min(128, lookback - 2, max(1, lookback // 2))
        values = set(range(1, min(32, cap) + 1))
        values.update(
            min(cap, max(1, value))
            for value in (
                short,
                seasonal_period // 2,
                seasonal_period,
                2 * seasonal_period,
                lookback // 3,
                cap,
            )
        )
        lags = tuple(sorted(values))
        if kind == "ridge":
            levels = (
                0.0,
                1e-6,
                1e-5,
                1e-4,
                0.001,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
                30.0,
                100.0,
                300.0,
                1000.0,
                3000.0,
                10000.0,
            )
            return _CandidateSpace(
                {"lags": short, "alpha": 1.0}, axes=(("lags", lags), ("alpha", levels))
            )
        baseline = {
            "lags": short,
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            "seed": seed,
        }
        return _CandidateSpace(
            baseline,
            axes=(
                ("lags", lags),
                ("n_estimators", (25, 50, 100, 200, 400)),
                ("max_depth", (1, 2, 3, 4, 6)),
                ("learning_rate", (0.02, 0.05, 0.1, 0.2)),
                ("reg_lambda", (0.0, 0.1, 1.0, 10.0, 100.0)),
            ),
            fixed={"seed": seed},
        )
    if kind == "chronos":
        base = dict(spec.get("params", {}))
        baseline = dict(base, seed=seed, temperature=1.0, num_samples=20)
        effective = {"top_k": 50, "top_p": 1.0, **baseline}
        return _CandidateSpace(
            baseline,
            axes=(
                ("num_samples", (10, 20, 40, 80)),
                ("temperature", (0.5, 0.7, 0.9, 1.0, 1.1, 1.3)),
                ("top_k", (20, 50, 100)),
                ("top_p", (0.8, 0.9, 0.95, 1.0)),
            ),
            fixed={"seed": seed},
            effective_baseline=effective,
        )
    raise ValueError(f"unsupported expert tuning kind: {kind!r}")


class ModelAssetMismatchError(ValueError):
    pass


def chronos_assets(folder: Path, expected_kind: str | None = None) -> dict[str, str]:
    config = folder / "config.json"
    if not config.is_file():
        raise ValueError("local Chronos config.json is missing")
    payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("chronos_config"), dict
    ):
        raise ModelAssetMismatchError(
            "local config is not a supported Chronos model configuration"
        )
    chronos = payload["chronos_config"]
    kind = "chronos_bolt" if "input_patch_size" in chronos else "chronos"
    if expected_kind is not None and kind != expected_kind:
        raise ModelAssetMismatchError(
            f"prepared model has kind {kind}, expected {expected_kind}"
        )
    for key in ("context_length", "prediction_length"):
        if (
            isinstance(chronos.get(key), bool)
            or not isinstance(chronos.get(key), int)
            or chronos[key] < 1
        ):
            raise ModelAssetMismatchError(f"Chronos config requires a positive {key}")
    if kind == "chronos_bolt":
        if payload.get(
            "chronos_pipeline_class"
        ) != "ChronosBoltPipeline" or payload.get("architectures") != [
            "ChronosBoltModelForForecasting"
        ]:
            raise ModelAssetMismatchError(
                "Chronos-Bolt config must declare its quantile pipeline and architecture"
            )
        quantiles = chronos.get("quantiles")
        if (
            not isinstance(quantiles, list)
            or not quantiles
            or any(
                isinstance(q, bool)
                or not isinstance(q, (int, float))
                or not math.isfinite(q)
                or not 0 < q < 1
                for q in quantiles
            )
            or quantiles != sorted(set(quantiles))
            or 0.5 not in quantiles
        ):
            raise ModelAssetMismatchError(
                "Chronos-Bolt config must include ordered quantiles and a median"
            )
    else:
        if (
            not isinstance(chronos.get("tokenizer_class"), str)
            or payload.get("model_type") != "t5"
        ):
            raise ModelAssetMismatchError(
                "Chronos-T5 config requires its T5 numeric tokenizer architecture"
            )
        if (
            payload.get("chronos_pipeline_class", "ChronosPipeline")
            != "ChronosPipeline"
        ):
            raise ModelAssetMismatchError(
                "Chronos-T5 config declares a different prediction pipeline"
            )
    weights = sorted(folder.glob("*.safetensors")) or sorted(
        folder.glob("pytorch_model*.bin")
    )
    if not weights or any(
        not path.is_file() or path.stat().st_size == 0 for path in weights
    ):
        raise ValueError("local Chronos weight files are missing or empty")
    for index in folder.glob("*.index.json"):
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        for shard in index_payload.get("weight_map", {}).values():
            path = (folder / shard).resolve()
            if path.parent != folder.resolve() or not path.is_file():
                raise ValueError(
                    "Chronos weight index contains a missing or unsafe shard"
                )
    paths = sorted(
        {
            config,
            *weights,
            *(
                path
                for path in folder.glob("*.json")
                if path.name != "prepared_model.json"
            ),
        }
    )
    return {path.name: file_sha256(path) for path in paths}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
