"""Causal expert preparation, variant retention, calibration, and confirmation."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anchorer.build import (
    AnchorLibrary,
    anchor_forecast_from_library,
    assign_library_cluster,
    build_anchor_library,
    extend_library_cases,
)
from anchorer.calibration import fit_library_calibration
from anchorer.quality import assess_quality, representative_origins, summarize_metrics
from anchorer.selection import retain_family_variants
from backends.catalog import (
    candidate_capacity,
    candidate_parameters,
    content_fingerprint,
    file_sha256,
    preflight_models,
    prepare_models,
    selected_model_names,
)
from backends.experts import finite_vector, forecast_expert_specs
from utils.io import DatasetView, atomic_write_json, load_dataset
from utils.metrics import mae, mse


@dataclass(slots=True)
class _PreparedDataset:
    spec: dict[str, Any]
    dataset: DatasetView
    fingerprint: str
    fit_end_row: int
    geometry_start: int
    split: int
    calibration_start: int
    confirmation_start: int
    origins: list[int]
    calibration_origins: list[int]
    confirmation_origins: list[int]
    global_fit_end_row: int | None = None


def _configuration(
    *,
    tuning_rounds=64,
    validation_fraction=None,
    seed=0,
    max_windows=None,
    tuning_windows=24,
    calibration_windows=24,
    confirmation_windows=24,
    min_confirmation_windows=4,
    min_calibration_windows=4,
    quality_policy="required",
    promotion_policy="none",
    min_mse_improvement=0.01,
    max_mae_regression=0.0,
    fit_fraction=0.60,
    selection_start_fraction=None,
    calibration_start_fraction=None,
    confirmation_start_fraction=None,
    retain_variants=3,
    loss_mode="normalized",
    model_device="cpu",
    dlinear_epochs=20,
    dlinear_variants=3,
    prediction_atol=1e-6,
    prediction_rtol=1e-5,
    prediction_correlation=0.999,
    prediction_relative_rmse=0.01,
):
    settings = locals().copy()
    for name in (
        "tuning_rounds",
        "tuning_windows",
        "calibration_windows",
        "confirmation_windows",
        "min_confirmation_windows",
        "min_calibration_windows",
        "dlinear_epochs",
        "dlinear_variants",
    ):
        _positive_integer(settings[name], name)
    _positive_integer(seed, "seed", 0)
    _positive_integer(retain_variants, "retain_variants", 0)
    if tuning_rounds > 256 or dlinear_variants > 3:
        raise ValueError(
            "tuning_rounds must not exceed 256 and DLinear variants must not exceed 3"
        )
    if max_windows is not None:
        _positive_integer(max_windows, "max_windows")
    if quality_policy not in {"required", "report"} or loss_mode not in {
        "normalized",
        "raw",
    }:
        raise ValueError(
            "quality_policy must be required/report and loss_mode normalized/raw"
        )
    if promotion_policy not in {"none", "conservative"}:
        raise ValueError("promotion_policy must be none or conservative")
    if quality_policy == "required" and (
        min_confirmation_windows < 4 or min_calibration_windows < 4
    ):
        raise ValueError(
            "required quality needs at least four calibration and confirmation windows"
        )
    for name in (
        "fit_fraction",
        "min_mse_improvement",
        "max_mae_regression",
        "prediction_atol",
        "prediction_rtol",
        "prediction_correlation",
        "prediction_relative_rmse",
    ):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
    if (
        not 0 < fit_fraction < 1
        or not 0 <= min_mse_improvement < 1
        or max_mae_regression < 0
    ):
        raise ValueError("invalid fit fraction or quality thresholds")
    if (
        prediction_atol < 0
        or prediction_rtol < 0
        or prediction_relative_rmse < 0
        or not -1 <= prediction_correlation <= 1
    ):
        raise ValueError("invalid prediction deduplication thresholds")
    explicit = (
        selection_start_fraction,
        calibration_start_fraction,
        confirmation_start_fraction,
    )
    if validation_fraction is not None:
        if any(value is not None for value in explicit):
            raise ValueError(
                "legacy validation_fraction cannot be mixed with explicit stage fractions"
            )
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not math.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1
        ):
            raise ValueError(
                "validation_fraction must be strictly between zero and one"
            )
        selection = 1 - validation_fraction
        confirmation = selection + validation_fraction / 2
        calibration = (selection + confirmation) / 2
    else:
        selection = (
            0.75 if selection_start_fraction is None else selection_start_fraction
        )
        calibration = (
            0.825 if calibration_start_fraction is None else calibration_start_fraction
        )
        confirmation = (
            0.90 if confirmation_start_fraction is None else confirmation_start_fraction
        )
    fractions = (selection, calibration, confirmation)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in fractions
        )
        or not 0 < selection < calibration < confirmation < 1
    ):
        raise ValueError(
            "stage fractions must satisfy 0<selection<calibration<confirmation<1"
        )
    if not isinstance(model_device, str) or not model_device.strip():
        raise ValueError("model_device must be an explicit device name")
    settings.update(
        selection_start_fraction=selection,
        calibration_start_fraction=calibration,
        confirmation_start_fraction=confirmation,
    )
    return settings


def _dataset_identity(item):
    return {
        "spec": item.spec,
        "sha256": item.fingerprint,
        "fit_end_row": item.fit_end_row,
        "geometry_start_row": item.geometry_start,
        "global_fit_end_row": item.global_fit_end_row,
        "geometry_end_row_exclusive": item.split,
        "selection_start": item.split,
        "calibration_start": item.calibration_start,
        "confirmation_start": item.confirmation_start,
        "selection_origins": item.origins,
        "calibration_origins": item.calibration_origins,
        "confirmation_origins": item.confirmation_origins,
    }


def _preflight_datasets(
    specs,
    validation_fraction=None,
    max_windows=None,
    *,
    tuning_windows=24,
    calibration_windows=24,
    confirmation_windows=24,
    min_confirmation_windows=4,
    min_calibration_windows=4,
    quality_policy="required",
    fit_fraction=0.60,
    promotion_policy="none",
    selection_start_fraction=0.75,
    calibration_start_fraction=0.825,
    confirmation_start_fraction=0.90,
    require_global_fit=False,
    **unused,
):
    if (
        isinstance(specs, (str, bytes, dict))
        or not isinstance(specs, Sequence)
        or not specs
    ):
        raise ValueError(
            "dataset_specs must be a nonempty sequence of training dataset objects"
        )
    prepared, errors, names = [], [], set()
    for raw in specs:
        try:
            if not isinstance(raw, dict):
                raise ValueError("dataset specification must be an object")
            name = raw.get("name")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or name in {".", ".."}
                or name.casefold() in names
            ):
                raise ValueError("dataset names must be safe, nonempty, and unique")
            names.add(name.casefold())
            if raw.get("split", "train") not in {"train", "train_val", "training"}:
                raise ValueError("initialization accepts training data only")
            source = Path(raw["data_path"]).expanduser().resolve(strict=True)
            if (
                not source.is_file()
                or source.suffix.casefold() != ".csv"
                or any(
                    re.search(r"(^|[_\-.])test($|[_\-.])", part.casefold())
                    for part in source.parts
                )
            ):
                raise ValueError(
                    "initialization requires a training CSV and rejects test-named inputs"
                )
            spec = {"name": name, "data_path": str(source)}
            for key in ("lookback", "horizon", "seasonal_period"):
                _positive_integer(
                    raw.get(key), f"{name}.{key}", 2 if key == "lookback" else 1
                )
                spec[key] = raw[key]
            spec["stride"] = (
                spec["horizon"] if raw.get("stride") is None else raw["stride"]
            )
            _positive_integer(spec["stride"], f"{name}.stride")
            for key in ("target_col", "timestamp_col"):
                if raw.get(key) is not None and (
                    not isinstance(raw[key], str) or not raw[key]
                ):
                    raise ValueError(f"{name}.{key} must be a column name")
            before = file_sha256(source)
            dataset = load_dataset(
                source, raw.get("target_col"), raw.get("timestamp_col")
            )
            if file_sha256(source) != before:
                raise ValueError("data changed during preflight")
            spec["target_col"], spec["timestamp_col"] = (
                dataset.target_col,
                dataset.timestamp_col,
            )
            if promotion_policy == "conservative" and dataset.timestamp_col is None:
                raise ValueError(
                    "conservative promotion requires chronological timestamps"
                )
            total = len(dataset.frame)
            fit_end = math.floor(total * fit_fraction)
            selection = math.floor(total * selection_start_fraction)
            calibration = math.floor(total * calibration_start_fraction)
            confirmation = math.floor(total * confirmation_start_fraction)
            earliest = max(spec["lookback"], fit_end if require_global_fit else 0)
            geometry_start = (
                spec["lookback"]
                + ((earliest - spec["lookback"] + spec["stride"] - 1) // spec["stride"])
                * spec["stride"]
            )
            if require_global_fit and fit_end >= selection:
                raise ValueError(
                    "global model fit prefix must precede geometry and recipe selection"
                )
            if geometry_start + spec["horizon"] > selection:
                raise ValueError("prefix is too short for out-of-sample geometry cases")
            limits = (tuning_windows, calibration_windows, confirmation_windows)
            selected = representative_origins(
                selection,
                calibration,
                spec["horizon"],
                spec["stride"],
                limits[0],
                grid_origin=spec["lookback"],
            )
            calibrated = representative_origins(
                calibration,
                confirmation,
                spec["horizon"],
                spec["stride"],
                limits[1],
                grid_origin=spec["lookback"],
            )
            confirmed = representative_origins(
                confirmation,
                total,
                spec["horizon"],
                spec["stride"],
                limits[2],
                grid_origin=spec["lookback"],
            )
            if not selected or not calibrated or not confirmed:
                raise ValueError(
                    "selection, calibration and confirmation each require a complete target window"
                )
            if quality_policy == "required" and (
                len(calibrated) < min_calibration_windows
                or len(confirmed) < min_confirmation_windows
            ):
                raise ValueError(
                    f"required quality needs at least {min_calibration_windows} calibration and {min_confirmation_windows} confirmation windows"
                )
            prepared.append(
                _PreparedDataset(
                    spec,
                    dataset,
                    before,
                    fit_end,
                    geometry_start,
                    selection,
                    calibration,
                    confirmation,
                    selected,
                    calibrated,
                    confirmed,
                    fit_end if require_global_fit else None,
                )
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(
                f"{raw.get('name', '<unnamed>') if isinstance(raw, dict) else '<invalid>'}: {exc}"
            )
    if errors:
        raise ValueError("dataset preflight failed:\n- " + "\n- ".join(errors))
    return prepared


def preflight_case_library_inputs(
    dataset_specs,
    *,
    models=None,
    model_store=None,
    download_models=False,
    **settings,
):
    """Validate planned boundaries and dependencies without model fitting or writes."""
    config = _configuration(**settings)
    names = selected_model_names(models)
    prepared = _preflight_datasets(
        dataset_specs, **config, require_global_fit="DLinear" in names
    )
    if not isinstance(download_models, bool):
        raise ValueError("download_models must be boolean")
    if model_store is None:
        if "DLinear" in names:
            raise ValueError(
                "model_store is required for DLinear preparation preflight"
            )
        for item in prepared:
            _check_lookback(item, names)
    else:
        _check_model_inputs(
            prepared,
            names,
            Path(model_store).expanduser().resolve(),
            config,
            download_models,
        )
    return {
        "models": names,
        "settings": config,
        "memory_bounds": {
            item.spec["name"]: {
                "start_origin_row": item.geometry_start,
                "end_row": item.confirmation_start,
            }
            for item in prepared
        },
        "datasets": [_dataset_identity(item) for item in prepared],
    }


def _check_lookback(item, names):
    if (
        any(name in {"Ridge", "XGBoost", "LightGBM"} for name in names)
        and item.spec["lookback"] < 3
    ):
        raise ValueError("autoregressive experts require lookback >= 3")
    if "AutoARIMA" in names and item.spec["lookback"] < 4:
        raise ValueError("AutoARIMA requires lookback >= 4")


def _check_model_inputs(prepared, names, store, settings, download_models):
    preflight_models(
        names, store, download_models=download_models, device=settings["model_device"]
    )
    for item in prepared:
        _check_lookback(item, names)
    if "DLinear" in names:
        from backends.neural import preflight_dlinear_inputs

        for item in prepared:
            preflight_dlinear_inputs(
                item.spec["data_path"],
                fit_end_row=item.fit_end_row,
                lookback=item.spec["lookback"],
                horizon=item.spec["horizon"],
                target_col=item.spec["target_col"],
                timestamp_col=item.spec["timestamp_col"],
                seed=settings["seed"],
                device=settings["model_device"],
                epochs=settings["dlinear_epochs"],
                variants=settings["dlinear_variants"],
            )


def initialize_case_libraries(
    dataset_specs: Sequence[dict[str, Any]],
    output_root: str | Path,
    model_store: str | Path,
    *,
    models: Sequence[str] | None = None,
    tuning_rounds: int = 64,
    validation_fraction: float | None = None,
    seed: int = 0,
    download_models: bool = False,
    max_windows: int | None = None,
    resume: bool = True,
    tuning_windows: int = 24,
    calibration_windows: int = 24,
    confirmation_windows: int = 24,
    min_confirmation_windows: int = 4,
    min_calibration_windows: int = 4,
    quality_policy: str = "required",
    promotion_policy: str = "none",
    min_mse_improvement: float = 0.01,
    max_mae_regression: float = 0.0,
    fit_fraction: float = 0.60,
    selection_start_fraction: float | None = None,
    calibration_start_fraction: float | None = None,
    confirmation_start_fraction: float | None = None,
    retain_variants: int = 3,
    loss_mode: str = "normalized",
    model_device: str = "cpu",
    dlinear_epochs: int = 20,
    dlinear_variants: int = 3,
    prediction_atol: float = 1e-6,
    prediction_rtol: float = 1e-5,
    prediction_correlation: float = 0.999,
    prediction_relative_rmse: float = 0.01,
) -> dict[str, Any]:
    """Prepare on a prefix, retain recipes, calibrate causally, then confirm once.

    The final bank keeps fixed geometry and historical calibrated losses while
    extending reference coverage only through the pre-confirmation prefix.
    """
    settings = _configuration(
        tuning_rounds=tuning_rounds,
        validation_fraction=validation_fraction,
        seed=seed,
        max_windows=max_windows,
        tuning_windows=tuning_windows,
        calibration_windows=calibration_windows,
        confirmation_windows=confirmation_windows,
        min_confirmation_windows=min_confirmation_windows,
        min_calibration_windows=min_calibration_windows,
        quality_policy=quality_policy,
        promotion_policy=promotion_policy,
        min_mse_improvement=min_mse_improvement,
        max_mae_regression=max_mae_regression,
        fit_fraction=fit_fraction,
        selection_start_fraction=selection_start_fraction,
        calibration_start_fraction=calibration_start_fraction,
        confirmation_start_fraction=confirmation_start_fraction,
        retain_variants=retain_variants,
        loss_mode=loss_mode,
        model_device=model_device,
        dlinear_epochs=dlinear_epochs,
        dlinear_variants=dlinear_variants,
        prediction_atol=prediction_atol,
        prediction_rtol=prediction_rtol,
        prediction_correlation=prediction_correlation,
        prediction_relative_rmse=prediction_relative_rmse,
    )
    if not isinstance(download_models, bool) or not isinstance(resume, bool):
        raise ValueError("download_models and resume must be boolean")
    names = selected_model_names(models)
    prepared = _preflight_datasets(
        dataset_specs, **settings, require_global_fit="DLinear" in names
    )
    output, store = (
        Path(output_root).expanduser().resolve(),
        Path(model_store).expanduser().resolve(),
    )
    if output.exists() and not output.is_dir():
        raise ValueError("output_root must be a directory")
    _check_model_inputs(prepared, names, store, settings, download_models)
    store.mkdir(parents=True, exist_ok=True)
    native_names = [name for name in names if name != "DLinear"]
    with _exclusive_lock(store / ".preparation.lock"):
        shared = (
            prepare_models(
                native_names,
                store,
                download_models=download_models,
                device=model_device,
            )
            if native_names
            else []
        )
        model_metadata = (
            preflight_models(native_names, store, download_models=False)
            if native_names
            else {}
        )
        experts = {}
        for item in prepared:
            specs = [
                dict(
                    spec,
                    family=spec.get("family", spec["name"]),
                    is_family_default=spec.get("is_family_default", True),
                )
                for spec in shared
            ]
            if "DLinear" in names:
                from backends.neural import prepare_dlinear_experts

                specs.extend(
                    prepare_dlinear_experts(
                        item.spec["data_path"],
                        store / item.spec["name"],
                        fit_end_row=item.fit_end_row,
                        lookback=item.spec["lookback"],
                        horizon=item.spec["horizon"],
                        target_col=item.spec["target_col"],
                        timestamp_col=item.spec["timestamp_col"],
                        seed=seed,
                        device=model_device,
                        epochs=dlinear_epochs,
                        variants=dlinear_variants,
                    )
                )
            cutoffs = [
                spec["fit_end_row"]
                for spec in specs
                if spec.get("fit_end_row") is not None
            ]
            for cutoff in cutoffs:
                _positive_integer(cutoff, "global model fit cutoff", 0)
            earliest = max([item.spec["lookback"], *cutoffs])
            actual = (
                item.spec["lookback"]
                + (
                    (earliest - item.spec["lookback"] + item.spec["stride"] - 1)
                    // item.spec["stride"]
                )
                * item.spec["stride"]
            )
            if actual + item.spec["horizon"] > item.split:
                raise ValueError(
                    "global model fit cutoff leaves no out-of-sample geometry cases"
                )
            item.geometry_start = actual
            item.global_fit_end_row = max(cutoffs) if cutoffs else None
            experts[item.spec["name"]] = specs
            _assert_unchanged(item)
    identity = {
        "schema_version": 3,
        "datasets": [_dataset_identity(item) for item in prepared],
        "experts": experts,
        "model_metadata": model_metadata,
        "settings": settings,
        "implementation": _implementation_identity(),
    }
    fingerprint = content_fingerprint(identity)
    suffix = (
        fingerprint[:20] if resume else f"{fingerprint[:20]}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = output / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_dir / ".initialization.lock"):
        old = _read_record(run_dir / "identity.json")
        if old is not None and old != identity:
            raise ValueError(
                "initialization identity conflicts with its output directory"
            )
        _write_record(run_dir / "identity.json", identity)
        _write_record(
            run_dir / "initialization.json",
            {"status": "running", "fingerprint": fingerprint},
        )
        outputs = {}
        try:
            for item in prepared:
                _assert_unchanged(item)
                directory = run_dir / item.spec["name"]
                directory.mkdir(parents=True, exist_ok=True)
                from backends.experts import arima_order_cache

                with arima_order_cache():
                    outputs[item.spec["name"]] = _initialize_dataset(
                        item,
                        experts[item.spec["name"]],
                        directory,
                        fingerprint=fingerprint,
                        **settings,
                    )
                _assert_unchanged(item)
            result = {
                "status": "complete",
                "fingerprint": fingerprint,
                "output_dir": str(run_dir),
                "model_store": str(store),
                "datasets": outputs,
                "quality_policy": quality_policy,
                "quality_passed": all(
                    info["quality_passed"] for info in outputs.values()
                ),
                "quality_scope": "frozen_foundational_anchorer_only",
            }
            if promotion_policy == "conservative":
                result.update(
                    promotion_policy=promotion_policy,
                    quality_scope="training_side_promotion_selection",
                    independent_quality_verified=False,
                )
            _write_record(run_dir / "initialization.json", result)
            return result
        except Exception as exc:
            _write_record(
                run_dir / "initialization.json",
                {
                    "status": "failed",
                    "fingerprint": fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise


def _history(item, origin):
    spec = item.spec
    history = (
        item.dataset.frame[spec["target_col"]]
        .iloc[origin - spec["lookback"] : origin]
        .to_numpy(dtype=float, copy=True)
    )
    timestamps = (
        item.dataset.frame[spec["timestamp_col"]]
        .iloc[origin - spec["lookback"] : origin]
        .to_numpy(copy=True)
        if spec["timestamp_col"]
        else None
    )
    return history, timestamps


def _score_origin(item, origin, identity, path, predictor):
    record = _read_record(path)
    if record is None:
        history, timestamps = _history(item, origin)
        try:
            prediction = finite_vector(
                predictor(history, timestamps), "selection forecast"
            )
            target = (
                item.dataset.frame[item.spec["target_col"]]
                .iloc[origin : origin + item.spec["horizon"]]
                .to_numpy(dtype=float)
            )
            record = {
                "identity": identity,
                "origin": origin,
                "status": "valid",
                "prediction": prediction.tolist(),
                "mse": mse(prediction, target),
                "mae": mae(prediction, target),
                "error": "",
            }
        except Exception as exc:
            record = {
                "identity": identity,
                "origin": origin,
                "status": "failed",
                "prediction": None,
                "mse": None,
                "mae": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_record(path, record)
    if (
        not isinstance(record, dict)
        or record.get("identity") != identity
        or record.get("origin") != origin
    ):
        raise ValueError(f"scoring checkpoint has incompatible identity: {path}")
    if record.get("status") == "valid":
        values = finite_vector(record.get("prediction"), "saved forecast")
        if len(values) != item.spec["horizon"] or any(
            isinstance(record.get(key), bool)
            or not isinstance(record.get(key), (int, float))
            or not math.isfinite(record[key])
            or record[key] < 0
            for key in ("mse", "mae")
        ):
            raise ValueError("scoring checkpoint contains invalid horizon or metrics")
    elif (
        record.get("status") != "failed"
        or not isinstance(record.get("error"), str)
        or not record["error"]
    ):
        raise ValueError("scoring checkpoint has an invalid failure record")
    return record


def _score_recipe(
    item, candidate, origins, directory, fingerprint, *, original_default=False
):
    records = []
    for origin in origins:
        identity = {
            "pipeline_fingerprint": fingerprint,
            "candidate": candidate,
            "origin": origin,
        }

        def predict(history, timestamps):
            predictions, _ = forecast_expert_specs(
                [candidate],
                history,
                item.spec["horizon"],
                item.spec["seasonal_period"],
                timestamps=timestamps,
            )
            return predictions[candidate["name"]]

        records.append(
            _score_origin(
                item, origin, identity, directory / f"origin_{origin:08d}.json", predict
            )
        )
    result = {
        "candidate": candidate,
        "candidate_id": content_fingerprint(candidate),
        "original_default": original_default,
        "origins": origins,
        "records": records,
        **summarize_metrics(records),
    }
    old = _read_record(directory / "trial.json")
    if old is not None and old != result:
        raise ValueError("candidate summary differs from its durable origin records")
    _write_record(directory / "trial.json", result)
    return result


def _select_recipes(item, experts, directory, fingerprint, settings):
    trials, capacities = [], {}
    for expert in experts:
        candidates = candidate_parameters(
            expert,
            lookback=item.spec["lookback"],
            horizon=item.spec["horizon"],
            seasonal_period=item.spec["seasonal_period"],
            tuning_rounds=settings["tuning_rounds"],
            seed=settings["seed"],
        )
        family = expert.get("family", expert["name"])
        capacities[family] = capacities.get(family, 0) + candidate_capacity(
            expert,
            lookback=item.spec["lookback"],
            horizon=item.spec["horizon"],
            seasonal_period=item.spec["seasonal_period"],
        )
        for number, params in enumerate(candidates):
            candidate = dict(expert, params=params)
            path = (
                directory
                / "selection"
                / expert["name"]
                / f"{number:03d}-{content_fingerprint(candidate)[:16]}"
            )
            trials.append(
                _score_recipe(
                    item,
                    candidate,
                    item.origins,
                    path,
                    fingerprint,
                    original_default=number == 0
                    and bool(expert.get("is_family_default", True)),
                )
            )
    try:
        selected, tuning, policy = retain_family_variants(
            trials,
            tuned_per_family=settings["retain_variants"],
            prediction_atol=settings["prediction_atol"],
            prediction_rtol=settings["prediction_rtol"],
            correlation_threshold=settings["prediction_correlation"],
            relative_rmse_threshold=settings["prediction_relative_rmse"],
        )
    except ValueError as exc:
        _write_record(
            directory / "selection_failed.json", {"error": str(exc), "trials": trials}
        )
        raise
    for summary in tuning:
        family_trials = [
            trial
            for trial in trials
            if trial["candidate"].get("family", trial["candidate"]["name"])
            == summary["family"]
        ]
        winner = min(
            (trial for trial in family_trials if trial["status"] == "valid"),
            key=lambda trial: (trial["mse"], trial["mae"], trials.index(trial)),
        )
        summary.update(
            model=summary["family"],
            selected=winner["candidate"]["params"],
            mse=winner["mse"],
            mae=winner["mae"],
            baseline_valid=summary["original_default_valid"],
            requested_trials=settings["tuning_rounds"],
            available_trials=capacities[summary["family"]],
            pool_exhausted=summary["attempted_trials"] >= capacities[summary["family"]],
            validation_windows=len(item.origins),
        )
    return selected, tuning, policy


def _protocol(item, tuning, retention, settings):
    timestamp_col = item.spec["timestamp_col"]
    timestamps = item.dataset.frame[timestamp_col] if timestamp_col else None
    return {
        "split": "training_csv_only",
        "source_sha256": item.fingerprint,
        "source_rows": len(item.dataset.frame),
        "chronology": "strict_timestamps" if timestamp_col else "supplied_row_order",
        "planned_fit_end_row": item.fit_end_row,
        "global_fit_end_row": item.global_fit_end_row,
        "geometry_start_row": item.geometry_start,
        "geometry_end_row_exclusive": item.split,
        "validation_start_row": item.split,
        "selection_start_row": item.split,
        "calibration_start_row": item.calibration_start,
        "confirmation_start_row": item.confirmation_start,
        "library_end_row_exclusive": item.confirmation_start,
        "memory_start_origin_row": item.geometry_start,
        "memory_end_row_exclusive": item.confirmation_start,
        "validation_origins": item.origins,
        "selection_origins": item.origins,
        "calibration_origins": item.calibration_origins,
        "confirmation_origins": item.confirmation_origins,
        "origin_grid_offset": item.spec["lookback"],
        "case_origin_stride": item.spec["stride"],
        "scoring_origin_stride": item.spec["stride"]
        * ((item.spec["horizon"] + item.spec["stride"] - 1) // item.spec["stride"]),
        "target_windows_overlap_between_stages": False,
        "tuning": "selection_only_rolling_origin_history_inputs",
        "final_library_scope": "fixed_geometry_plus_preconfirmation_references",
        "library_window_limit": settings["max_windows"],
        "historical_target_cutoff": (
            str(timestamps.iloc[item.confirmation_start - 1])
            if timestamps is not None
            else item.confirmation_start - 1
        ),
        "independent_test_evaluation": False,
        "default_incumbent_selected_on": "selection_MSE_then_MAE",
        "default_incumbent": retention["incumbent_recipe"],
        "default_incumbent_name": retention["incumbent_name"],
        "retention": retention,
        "loss_mode": settings["loss_mode"],
        "quality_scope": "frozen_foundational_anchorer_only",
    }


def _calibration_records(
    item, library, selected, incumbent_name, directory, fingerprint
):
    geometry_hash = content_fingerprint(library.to_dict())
    seasonal = {
        "name": "seasonal_naive_reference",
        "kind": "seasonal_naive",
        "params": {"season_length": item.spec["seasonal_period"]},
    }
    records = []
    for origin in item.calibration_origins:
        identity = {
            "pipeline_fingerprint": fingerprint,
            "origin": origin,
            "geometry_sha256": geometry_hash,
            "experts": selected,
            "incumbent_name": incumbent_name,
        }
        path = directory / f"origin_{origin:08d}.json"
        record = _read_record(path)
        if record is None:
            history, timestamps = _history(item, origin)
            try:
                predictions, _ = forecast_expert_specs(
                    selected,
                    history,
                    item.spec["horizon"],
                    item.spec["seasonal_period"],
                    timestamps=timestamps,
                )
                naive, _ = forecast_expert_specs(
                    [seasonal],
                    history,
                    item.spec["horizon"],
                    item.spec["seasonal_period"],
                    timestamps=timestamps,
                )
                # Every forecast above sees history only; labels are attached afterwards.
                target = (
                    item.dataset.frame[item.spec["target_col"]]
                    .iloc[origin : origin + item.spec["horizon"]]
                    .to_numpy(dtype=float, copy=True)
                )
                record = {
                    "identity": identity,
                    "status": "valid",
                    "origin": origin,
                    "target_end_row": origin + item.spec["horizon"],
                    "cluster_id": assign_library_cluster(library, history),
                    "predictions": {
                        name: finite_vector(value, name).tolist()
                        for name, value in predictions.items()
                    },
                    "target": target.tolist(),
                    "naive_prediction": finite_vector(
                        naive[seasonal["name"]], "seasonal reference"
                    ).tolist(),
                    "incumbent_prediction": finite_vector(
                        predictions[incumbent_name], "fixed incumbent"
                    ).tolist(),
                }
            except Exception as exc:
                record = {
                    "identity": identity,
                    "status": "failed",
                    "origin": origin,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _write_record(path, record)
        if (
            not isinstance(record, dict)
            or record.get("identity") != identity
            or record.get("origin") != origin
        ):
            raise ValueError("calibration checkpoint has incompatible identity")
        if record.get("status") not in {"valid", "failed"}:
            raise ValueError("calibration checkpoint has an invalid status")
        records.append(record)
    failed = [record["origin"] for record in records if record["status"] != "valid"]
    if failed:
        raise ValueError(
            f"calibration requires every preselected origin; failed origins: {failed}"
        )
    return [
        {
            key: value
            for key, value in record.items()
            if key not in {"identity", "status"}
        }
        for record in records
    ]


def _load_library_artifact(path, checkpoint, fingerprint, manifest_path):
    record = _read_record(checkpoint)
    if record is None:
        return None
    if (
        record.get("pipeline_fingerprint") != fingerprint
        or record.get("manifest_sha256") != file_sha256(manifest_path)
        or not path.is_file()
        or record.get("library_sha256") != file_sha256(path)
    ):
        raise ValueError("completed library configuration or contents changed")
    return AnchorLibrary.load(path)


def _save_library_artifact(
    library, path, checkpoint, fingerprint, manifest_path, protocol
):
    payload = library.to_dict()
    payload["initialization"] = {"pipeline_fingerprint": fingerprint, **protocol}
    atomic_write_json(path, payload)
    _write_record(
        checkpoint,
        {
            "pipeline_fingerprint": fingerprint,
            "manifest_sha256": file_sha256(manifest_path),
            "library_sha256": file_sha256(path),
            "case_count": len(library.cases),
        },
    )
    return AnchorLibrary.load(path)


def _prepare_library(
    item, selected, manifest_path, directory, fingerprint, protocol, settings
):
    spec = item.spec
    path = directory / "anchor_library.json"
    checkpoint = directory / "library_complete.json"
    library = _load_library_artifact(path, checkpoint, fingerprint, manifest_path)
    if library is None:
        geometry_path = directory / "geometry_library.json"
        geometry_complete = directory / "geometry_complete.json"
        library = _load_library_artifact(
            geometry_path, geometry_complete, fingerprint, manifest_path
        )
        if library is None:
            cap = settings["max_windows"]
            geometry_cap = (
                None
                if cap is None
                else (1 if cap == 1 else min(cap - 1, max(1, math.ceil(cap * 0.7))))
            )
            library = build_anchor_library(
                spec["data_path"],
                lookback=spec["lookback"],
                horizon=spec["horizon"],
                seasonal_period=spec["seasonal_period"],
                dataset_name=spec["name"],
                target_col=spec["target_col"],
                timestamp_col=spec["timestamp_col"],
                stride=spec["stride"],
                max_windows=geometry_cap,
                expert_manifest=manifest_path,
                start_row=item.geometry_start,
                end_row=item.split,
                checkpoint_dir=directory / "geometry_case_checkpoints",
            )
            _assert_unchanged(item)
            library = _save_library_artifact(
                library,
                geometry_path,
                geometry_complete,
                fingerprint,
                manifest_path,
                protocol,
            )
        records = _calibration_records(
            item,
            library,
            selected,
            protocol["default_incumbent_name"],
            directory / "calibration",
            fingerprint,
        )
        calibration_options = (
            {
                "normalize_losses": False,
                "temperatures": (1,),
                "shrinkages": (0,),
                "active_family_limits": (None,),
            }
            if settings["loss_mode"] == "raw"
            else {}
        )
        library = fit_library_calibration(
            library,
            records,
            source_sha256=item.fingerprint,
            calibration_start_row=item.calibration_start,
            calibration_end_row=item.confirmation_start,
            confirmation_start_row=item.confirmation_start,
            incumbent_name=protocol["default_incumbent_name"],
            allow_unqualified=settings["quality_policy"] == "report",
            **calibration_options,
        )
        cap = settings["max_windows"]
        if cap is None or len(library.cases) < cap:
            library = extend_library_cases(
                library,
                spec["data_path"],
                start_row=item.geometry_start,
                end_row=item.confirmation_start,
                stride=spec["stride"],
                max_windows=cap,
                checkpoint_dir=directory / "reference_case_checkpoints",
            )
        _assert_unchanged(item)
        library = _save_library_artifact(
            library,
            path,
            checkpoint,
            fingerprint,
            manifest_path,
            protocol,
        )
    if library.expert_specs != selected:
        raise ValueError("completed library does not use the retained expert recipes")
    return library, path


def _initialize_dataset(item, experts, directory, *, fingerprint, **settings):
    selected, tuning, retention = _select_recipes(
        item, experts, directory, fingerprint, settings
    )
    protocol = _protocol(item, tuning, retention, settings)
    manifest = {"experts": selected, "selection": tuning, "protocol": protocol}
    _write_record(directory / "selection.json", manifest)
    manifest_path = directory / "expert_manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if existing is not None and existing != manifest:
        raise ValueError(
            "selected expert manifest differs from its durable selection records"
        )
    atomic_write_json(manifest_path, manifest)
    library, library_path = _prepare_library(
        item,
        selected,
        manifest_path,
        directory,
        fingerprint,
        protocol,
        settings,
    )
    if settings["promotion_policy"] == "conservative":
        return _initialize_promoted_dataset(
            item,
            library,
            library_path,
            selected,
            tuning,
            retention,
            protocol,
            manifest_path,
            directory,
            fingerprint,
            settings,
        )
    report_path = directory / "quality_report.json"
    quality_complete = _read_record(directory / "quality_complete.json")
    if quality_complete is not None:
        if (
            quality_complete.get("library_sha256") != file_sha256(library_path)
            or not report_path.is_file()
            or quality_complete.get("report_sha256") != file_sha256(report_path)
        ):
            raise ValueError(
                "completed quality report or frozen library is missing or changed"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _confirm_library(
            item,
            library,
            retention["incumbent_recipe"],
            directory / "confirmation",
            fingerprint,
        )
        decision = assess_quality(
            report["candidate"],
            report["seasonal_naive"],
            report["default_incumbent"],
            minimum_windows=settings["min_confirmation_windows"],
            min_mse_improvement=settings["min_mse_improvement"],
            max_mae_regression=settings["max_mae_regression"],
        )
        decision["checks"]["minimum_calibration_windows"] = (
            len(item.calibration_origins) >= settings["min_calibration_windows"]
        )
        decision["quality_passed"] = all(decision["checks"].values())
        decision["failed_checks"] = [
            name for name, passed in decision["checks"].items() if not passed
        ]
        decision["requirements"]["min_calibration_windows"] = settings[
            "min_calibration_windows"
        ]
        report.update(
            decision,
            protocol=protocol,
            selection=tuning,
            retention=retention,
            coverage=library.coverage,
            calibration=library.calibration,
            quality_policy=settings["quality_policy"],
            library_sha256=file_sha256(library_path),
            pipeline_fingerprint=fingerprint,
            scope="Anchorer confirmation on a reserved training-side tail; no learned-model or historical-paper reproduction claim",
        )
        atomic_write_json(report_path, report)
        _write_record(
            directory / "quality_complete.json",
            {
                "library_sha256": file_sha256(library_path),
                "report_sha256": file_sha256(report_path),
            },
        )
    if settings["quality_policy"] == "required" and not report["quality_passed"]:
        raise ValueError(
            f"{item.spec['name']}: frozen Anchorer failed required confirmation quality "
            f"({', '.join(report['failed_checks'])}); see {report_path}"
        )
    return {
        "anchor_library": str(library_path),
        "expert_manifest": str(manifest_path),
        "case_count": len(library.cases),
        "cluster_count": len(library.clusters),
        "selection": tuning,
        "retention": retention,
        "protocol": protocol,
        "coverage": library.coverage,
        "calibration": library.calibration,
        "quality_passed": report["quality_passed"],
        "quality_report": str(report_path),
        "quality": {
            key: report[key]
            for key in (
                "candidate",
                "seasonal_naive",
                "default_incumbent",
                "quality_passed",
                "failed_checks",
                "requirements",
                "scope",
            )
        },
    }


def _initialize_promoted_dataset(
    item,
    library,
    library_path,
    selected,
    tuning,
    retention,
    protocol,
    manifest_path,
    directory,
    fingerprint,
    settings,
):
    from anchorer.promotion import select_promoted_library

    geometry = AnchorLibrary.load(directory / "geometry_library.json")
    records = _calibration_records(
        item,
        geometry,
        selected,
        retention["incumbent_name"],
        directory / "calibration",
        fingerprint,
    )
    requirements = {
        key: settings[key]
        for key in (
            "min_confirmation_windows",
            "min_calibration_windows",
            "min_mse_improvement",
            "max_mae_regression",
        )
    }

    def evaluate(candidate, checkpoint_dir):
        report = _confirm_library(
            item, candidate, retention["incumbent_recipe"], checkpoint_dir, fingerprint
        )
        decision = assess_quality(
            report["candidate"],
            report["seasonal_naive"],
            report["default_incumbent"],
            minimum_windows=settings["min_confirmation_windows"],
            min_mse_improvement=settings["min_mse_improvement"],
            max_mae_regression=settings["max_mae_regression"],
        )
        decision["checks"]["minimum_calibration_windows"] = (
            len(item.calibration_origins) >= settings["min_calibration_windows"]
        )
        decision["quality_passed"] = all(decision["checks"].values())
        decision["failed_checks"] = [
            name for name, passed in decision["checks"].items() if not passed
        ]
        decision["requirements"]["min_calibration_windows"] = settings[
            "min_calibration_windows"
        ]
        report.update(
            decision,
            protocol=protocol,
            selection=tuning,
            retention=retention,
            coverage=candidate.coverage,
            calibration=candidate.calibration,
            quality_policy=settings["quality_policy"],
            pipeline_fingerprint=fingerprint,
        )
        return report

    promotion_end = max(item.confirmation_origins) + item.spec["horizon"]
    timestamp = item.spec["timestamp_col"]
    last_timestamp = str(item.dataset.frame[timestamp].iloc[promotion_end - 1])
    promoted = select_promoted_library(
        library,
        library_path,
        directory,
        calibration_records=records,
        selection_incumbent_name=retention["incumbent_name"],
        pipeline_fingerprint=fingerprint,
        protocol=protocol,
        requirements=requirements,
        evaluate=evaluate,
        quality_policy=settings["quality_policy"],
        promotion_end_row_exclusive=promotion_end,
        promotion_last_timestamp=last_timestamp,
    )
    report = promoted["report"]
    return {
        "anchor_library": promoted["anchor_library"],
        "expert_manifest": str(manifest_path),
        "prepared_anchor_library": str(library_path),
        "prepared_anchor_library_sha256": file_sha256(library_path),
        "memory_anchor_library": str(library_path),
        "memory_anchor_library_sha256": file_sha256(library_path),
        "anchor_library_sha256": file_sha256(promoted["anchor_library"]),
        "prepared_quality_report": promoted["prepared_quality_report"],
        "case_count": len(library.cases),
        "cluster_count": len(library.clusters),
        "selection": tuning,
        "retention": retention,
        "protocol": protocol,
        "coverage": library.coverage,
        "calibration": library.calibration,
        "quality_passed": report["quality_passed"],
        "quality_report": promoted["quality_report"],
        "promotion": promoted["promotion"],
        "artifact_role": promoted["artifact_role"],
        "independent_quality_verified": False,
        "quality": {
            key: report[key]
            for key in (
                "candidate",
                "seasonal_naive",
                "default_incumbent",
                "quality_passed",
                "failed_checks",
                "requirements",
                "scope",
                "independent_quality_verified",
            )
        },
    }


def _confirm_library(item, library, incumbent, directory, fingerprint):
    seasonal = {
        "name": "seasonal_naive_reference",
        "kind": "seasonal_naive",
        "params": {"season_length": item.spec["seasonal_period"]},
    }
    records = {"candidate": [], "seasonal_naive": [], "default_incumbent": []}
    for origin in item.confirmation_origins:
        for label in records:
            identity = {
                "pipeline_fingerprint": fingerprint,
                "library_contents_sha256": content_fingerprint(library.to_dict()),
                "origin": origin,
                "role": label,
                "recipe": (
                    incumbent
                    if label == "default_incumbent"
                    else (
                        seasonal
                        if label == "seasonal_naive"
                        else "frozen_saved_library"
                    )
                ),
            }

            def predict(history, timestamps, label=label):
                if label == "candidate":
                    return anchor_forecast_from_library(
                        history,
                        item.spec["horizon"],
                        item.spec["seasonal_period"],
                        library,
                        timestamps=timestamps,
                    )["ensemble_forecast_baseline"]
                recipe = incumbent if label == "default_incumbent" else seasonal
                predictions, _ = forecast_expert_specs(
                    [recipe],
                    history,
                    item.spec["horizon"],
                    item.spec["seasonal_period"],
                    timestamps=timestamps,
                )
                return predictions[recipe["name"]]

            path = directory / label / f"origin_{origin:08d}.json"
            records[label].append(_score_origin(item, origin, identity, path, predict))
    return {
        **{label: summarize_metrics(values) for label, values in records.items()},
        "origin_records": records,
        "origins": item.confirmation_origins,
        "default_incumbent_recipe": incumbent,
        "seasonal_naive_recipe": seasonal,
    }


def _positive_integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _assert_unchanged(item: _PreparedDataset) -> None:
    if file_sha256(item.spec["data_path"]) != item.fingerprint:
        raise ValueError(
            f"training data changed during initialization: {item.spec['name']}"
        )


def _write_record(path: Path, payload: Any) -> None:
    atomic_write_json(
        path, {"payload": payload, "sha256": content_fingerprint(payload)}
    )


def _read_record(path: Path) -> Any:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or "payload" not in record
        or record.get("sha256") != content_fingerprint(record["payload"])
    ):
        raise ValueError(f"initialization record failed checksum validation: {path}")
    return record["payload"]


def _implementation_identity() -> dict[str, Any]:
    from backends import catalog, experts, neural
    from anchorer import build, quality, calibration, selection, promotion, policies
    from utils import io, metrics

    modules = {
        "initialization": __file__,
        "catalog": catalog.__file__,
        "experts": experts.__file__,
        "library": build.__file__,
        "quality": quality.__file__,
        "calibration": calibration.__file__,
        "selection": selection.__file__,
        "promotion": promotion.__file__,
        "policies": policies.__file__,
        "neural": neural.__file__,
        "io": io.__file__,
        "metrics": metrics.__file__,
    }
    packages = {}
    for package in ("numpy", "pandas", "statsmodels", "dtaidistance"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    return {
        "python": sys.version,
        "packages": packages,
        "sources": {name: file_sha256(path) for name, path in modules.items()},
    }


@contextmanager
def _exclusive_lock(path: Path):
    """Use an OS lock released automatically on normal exit or process termination."""
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"another initialization process holds {path}"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"another initialization process holds {path}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
