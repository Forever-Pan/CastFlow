"""Named local expert configurations and explicitly requested model preparation."""

from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from backends.experts import validate_expert_specs
from utils.io import atomic_write_json


DEFAULT_MODELS = ("SeasonalNaive", "HistoricAverage", "AutoARIMA", "Theta")
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
}
_PACKAGES = {"chronos": "chronos-forecasting", "huggingface_hub": "huggingface-hub"}


def list_model_catalog() -> list[dict[str, Any]]:
    """Return capabilities without preparing assets, importing models, or using network."""
    return [
        dict(
            name=name,
            **copy.deepcopy(entry),
            default=name in DEFAULT_MODELS,
            local_weights_required=entry["kind"] == "chronos",
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
    names: Sequence[str], model_store: str | Path, *, download_models: bool = False
) -> dict[str, Any]:
    """Check all requested dependencies and local assets before any model is loaded."""
    store = Path(model_store).expanduser().resolve()
    if store.exists() and not store.is_dir():
        raise ValueError(f"model_store is not a directory: {store}")
    details = {}
    errors = []
    for name in selected_model_names(names):
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
        if entry["kind"] == "chronos":
            try:
                assets = chronos_assets(folder)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                if not download_models:
                    errors.append(
                        f"{name}: {exc}. Prepare weights at {folder} or explicitly enable --download-models"
                    )
        details[name] = {"dependencies": packages, "existing_assets": assets}
    if errors:
        raise ValueError("model preflight failed:\n- " + "\n- ".join(errors))
    return details


def prepare_models(
    names: Sequence[str], model_store: str | Path, *, download_models: bool = False
) -> list[dict[str, Any]]:
    """Persist model recipes; download pretrained weights only with explicit permission."""
    names = selected_model_names(names)
    details = preflight_models(names, model_store, download_models=download_models)
    store = Path(model_store).expanduser().resolve()
    result = []
    for name in names:
        entry = _CATALOG[name]
        folder = store / name
        folder.mkdir(parents=True, exist_ok=True)
        spec: dict[str, Any] = {"name": name, "kind": entry["kind"]}
        assets = details[name]["existing_assets"]
        if entry["kind"] == "chronos":
            if not assets:
                if not download_models:
                    raise FileNotFoundError(
                        f"pretrained weights not available at {folder}"
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
                        ],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not prepare {name} from {entry['repository']}. "
                        f"Check repository access or place its complete weights at {folder}, then resume."
                    ) from exc
                assets = chronos_assets(folder)
            spec["model_path"] = str(folder)
            spec["model_identity"] = content_fingerprint(assets)
            spec["params"] = {
                "num_samples": 20,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "seed": 0,
            }
        record = {
            "name": name,
            "fit_scope": entry["fit_scope"],
            "expert": spec,
            "dependencies": details[name]["dependencies"],
            "assets": assets,
        }
        if "repository" in entry:
            record["repository"] = entry["repository"]
        atomic_write_json(folder / "prepared_model.json", record)
        result.append(spec)
    return validate_expert_specs(result)


def candidate_parameters(
    spec: dict[str, Any],
    *,
    lookback: int,
    seasonal_period: int,
    tuning_rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return deterministic, bounded candidates; their selected parameters are saved verbatim."""
    kind = spec["kind"]
    lags = min(12, max(1, lookback // 4))
    short = max(1, min(lags, lookback - 2))
    base = dict(spec.get("params", {}))
    pools: dict[str, list[dict[str, Any]]] = {
        "seasonal_naive": [
            {"season_length": seasonal_period},
            {"season_length": max(1, seasonal_period // 2)},
            {"season_length": 2 * seasonal_period},
            {"season_length": 1},
        ],
        "historic_average": [
            {"window": lookback},
            {"window": min(lookback, seasonal_period)},
            {"window": max(1, lookback // 2)},
            {"window": min(lookback, 2 * seasonal_period)},
        ],
        "auto_arima": [
            {"max_p": 2, "max_q": 2, "d_values": [0, 1]},
            {"max_p": 1, "max_q": 1, "d_values": [0]},
            {"max_p": 1, "max_q": 1, "d_values": [1]},
            {"max_p": 3, "max_q": 1, "d_values": [0, 1]},
        ],
        "theta": [
            {"deseasonalize": True, "period": seasonal_period},
            {"deseasonalize": False, "period": seasonal_period},
            {"deseasonalize": True, "period": max(1, seasonal_period // 2)},
        ],
        "zero": [{}],
        "ridge": [
            {"lags": short, "alpha": 1.0},
            {"lags": short, "alpha": 0.1},
            {"lags": max(1, short // 2), "alpha": 10.0},
            {"lags": min(2 * short, lookback - 2), "alpha": 1.0},
        ],
        "xgboost": [
            {
                "lags": short,
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "reg_lambda": 1.0,
                "seed": seed,
            }
            for n, d, lr in [
                (100, 3, 0.05),
                (50, 2, 0.1),
                (150, 4, 0.03),
                (200, 2, 0.05),
            ]
        ],
        "lightgbm": [
            {
                "lags": short,
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "reg_lambda": 1.0,
                "seed": seed,
            }
            for n, d, lr in [
                (100, 3, 0.05),
                (50, 2, 0.1),
                (150, 4, 0.03),
                (200, 2, 0.05),
            ]
        ],
        "chronos": [
            dict(base, seed=seed, temperature=t, num_samples=n)
            for t, n in [(1.0, 20), (0.8, 20), (1.0, 40), (1.2, 20)]
        ],
    }
    candidates = []
    for params in pools[kind]:
        if params not in candidates:
            candidates.append(params)
    # The default candidate is always first; later rounds use a reproducible order.
    import numpy as np

    if len(candidates) > 2:
        rng = np.random.default_rng(seed)
        tail = [
            candidates[int(i)] for i in rng.permutation(np.arange(1, len(candidates)))
        ]
        candidates = candidates[:1] + tail
    return candidates[:tuning_rounds]


def chronos_assets(folder: Path) -> dict[str, str]:
    config = folder / "config.json"
    if not config.is_file():
        raise ValueError("local Chronos config.json is missing")
    payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("chronos_config"), dict
    ):
        raise ValueError("local config is not a Chronos-T5 model configuration")
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
