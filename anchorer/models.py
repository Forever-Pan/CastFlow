from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DATASET_PACKAGES = {
    "MOPEX": "castmaster",
    "mopex": "castmaster",
    "WP": "castmaster_windy",
    "windy": "castmaster_windy",
    "windy_power": "castmaster_windy",
    "power": "castmaster_windy",
    "SP": "castmaster_sunny",
    "sunny": "castmaster_sunny",
    "sunny_power": "castmaster_sunny",
    "BE": "castmaster_epf_be",
    "EPF_BE": "castmaster_epf_be",
    "DE": "castmaster_epf_de",
    "EPF_DE": "castmaster_epf_de",
    "FR": "castmaster_epf_fr",
    "EPF_FR": "castmaster_epf_fr",
    "NP": "castmaster_epf_np",
    "EPF_NP": "castmaster_epf_np",
    "PJM": "castmaster_epf_pjm",
    "EPF_PJM": "castmaster_epf_pjm",
    "ETTh1": "castmaster_etth1",
    "ETT_ETTh1": "castmaster_etth1",
    "ETTm1": "castmaster_ettm1",
    "ETT_ETTm1": "castmaster_ettm1",
}
DATASET_PACKAGES.update({key.lower(): value for key, value in list(DATASET_PACKAGES.items())})

ANCHORER_PACKAGE_ROOT = "backends.packages"


@dataclass(slots=True)
class AnchorerForecastResult:
    forecasts: dict[str, np.ndarray]
    failures: dict[str, str]
    model_names: list[str]


def anchorer_package_name(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    return DATASET_PACKAGES.get(dataset_name, DATASET_PACKAGES.get(dataset_name.lower()))


def import_anchorer_modules(dataset_name: str | None) -> tuple[Any, Any] | None:
    package = anchorer_package_name(dataset_name)
    if not package:
        return None
    module_prefix = f"{ANCHORER_PACKAGE_ROOT}.{package}"
    models_base = importlib.import_module(f"{module_prefix}.models.base")
    forecast_mod = importlib.import_module(f"{module_prefix}.tools.forecast")
    return models_base, forecast_mod


def get_anchorer_model_names(dataset_name: str | None) -> list[str]:
    modules = import_anchorer_modules(dataset_name)
    if modules is None:
        return []
    models_base, _ = modules
    return [str(model.alias) for model in models_base.get_default_models()]


def forecast_with_anchorer_models(
    dataset_name: str | None,
    history: np.ndarray,
    horizon: int,
    seasonal_period: int,
    timestamps: pd.Series | None = None,
    model_names: list[str] | None = None,
    strict: bool = False,
) -> AnchorerForecastResult:
    modules = import_anchorer_modules(dataset_name)
    if modules is None:
        return AnchorerForecastResult(forecasts={}, failures={"backends": "dataset package not found"}, model_names=[])
    models_base, forecast_mod = modules
    aliases = model_names or [str(model.alias) for model in models_base.get_default_models()]
    forecasts: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for alias in aliases:
        try:
            pred = forecast_mod.forecast_with_model_retry(
                alias,
                np.asarray(history, dtype=float),
                horizon,
                season_length=seasonal_period,
                timestamps=timestamps,
            )
            forecasts[alias] = np.asarray(pred, dtype=float)
        except Exception as exc:  # noqa: BLE001 - preserve model failure for auditability
            if strict:
                raise RuntimeError(f"anchorer model failed during case-library construction: {alias}") from exc
            failures[alias] = f"{type(exc).__name__}: {exc}"
    return AnchorerForecastResult(forecasts=forecasts, failures=failures, model_names=aliases)
