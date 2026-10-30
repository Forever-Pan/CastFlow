"""Resumable expert selection and held-out confirmation of a frozen anchor library."""

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
    build_anchor_library,
)
from anchorer.quality import assess_quality, representative_origins, summarize_metrics
from backends.catalog import (
    candidate_capacity,
    candidate_parameters,
    content_fingerprint,
    file_sha256,
    preflight_models,
    prepare_models,
    selected_model_names,
)
from backends.experts import forecast_expert_specs
from utils.io import DatasetView, atomic_write_json, load_dataset
from utils.metrics import mae, mse


@dataclass(slots=True)
class _PreparedDataset:
    spec: dict[str, Any]
    dataset: DatasetView
    fingerprint: str
    split: int
    confirmation_start: int
    origins: list[int]
    confirmation_origins: list[int]


def initialize_case_libraries(
    dataset_specs: Sequence[dict[str, Any]],
    output_root: str | Path,
    model_store: str | Path,
    *,
    models: Sequence[str] | None = None,
    tuning_rounds: int = 24,
    validation_fraction: float = 0.2,
    seed: int = 0,
    download_models: bool = False,
    max_windows: int | None = None,
    resume: bool = True,
    tuning_windows: int = 12,
    confirmation_windows: int = 12,
    min_confirmation_windows: int = 4,
    quality_policy: str = "required",
    min_mse_improvement: float = 0.01,
    max_mae_regression: float = 0.0,
) -> dict[str, Any]:
    """Select on the first training tail and confirm a frozen prefix library on the second.

    The two tail intervals are fixed chronologically before any forecast. Confirmation
    labels enter only the final scorer. The saved library stops at confirmation_start
    and is never refitted with its holdout. This assesses the Anchorer, not a learned
    Forecasting checkpoint or the paper's historical experiments.
    """
    for name, value in (
        ("tuning_rounds", tuning_rounds),
        ("tuning_windows", tuning_windows),
        ("confirmation_windows", confirmation_windows),
        ("min_confirmation_windows", min_confirmation_windows),
    ):
        _positive_integer(value, name)
    if tuning_rounds > 256:
        raise ValueError("tuning_rounds must not exceed 256 distinct attempted recipes")
    _positive_integer(seed, "seed", minimum=0)
    if quality_policy not in {"required", "report"}:
        raise ValueError("quality_policy must be required or report")
    if quality_policy == "required" and min_confirmation_windows < 4:
        raise ValueError("required quality needs at least four confirmation windows")
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("min_mse_improvement", min_mse_improvement),
        ("max_mae_regression", max_mae_regression),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
    if (
        not 0 < validation_fraction < 1
        or not 0 <= min_mse_improvement < 1
        or max_mae_regression < 0
    ):
        raise ValueError("invalid split fraction or quality comparison thresholds")
    if not isinstance(download_models, bool) or not isinstance(resume, bool):
        raise ValueError("download_models and resume must be boolean")
    if max_windows is not None:
        _positive_integer(max_windows, "max_windows")
    settings = dict(
        tuning_rounds=tuning_rounds,
        tuning_windows=tuning_windows,
        confirmation_windows=confirmation_windows,
        min_confirmation_windows=min_confirmation_windows,
        quality_policy=quality_policy,
        min_mse_improvement=min_mse_improvement,
        max_mae_regression=max_mae_regression,
        seed=seed,
        max_windows=max_windows,
    )
    names = selected_model_names(models)
    prepared = _preflight_datasets(
        dataset_specs,
        validation_fraction,
        max_windows,
        tuning_windows=tuning_windows,
        confirmation_windows=confirmation_windows,
        min_confirmation_windows=min_confirmation_windows,
        quality_policy=quality_policy,
    )
    for item in prepared:
        if (
            any(name in {"Ridge", "XGBoost", "LightGBM"} for name in names)
            and item.spec["lookback"] < 3
        ):
            raise ValueError("autoregressive experts require lookback >= 3")
        if "AutoARIMA" in names and item.spec["lookback"] < 4:
            raise ValueError("AutoARIMA requires lookback >= 4")
    output, store = (
        Path(output_root).expanduser().resolve(),
        Path(model_store).expanduser().resolve(),
    )
    if output.exists() and not output.is_dir():
        raise ValueError(f"output_root is not a directory: {output}")
    preflight_models(names, store, download_models=download_models)
    store.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(store / ".preparation.lock"):
        experts = prepare_models(names, store, download_models=download_models)
        model_metadata = preflight_models(names, store, download_models=False)
    identity = {
        "schema_version": 2,
        "datasets": [
            {
                "spec": item.spec,
                "sha256": item.fingerprint,
                "selection_start": item.split,
                "confirmation_start": item.confirmation_start,
                "selection_origins": item.origins,
                "confirmation_origins": item.confirmation_origins,
            }
            for item in prepared
        ],
        "experts": experts,
        "model_metadata": model_metadata,
        "settings": settings,
        "validation_fraction": validation_fraction,
        "implementation": _implementation_identity(),
    }
    fingerprint = content_fingerprint(identity)
    suffix = (
        fingerprint[:20] if resume else f"{fingerprint[:20]}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = output / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(run_dir / ".initialization.lock"):
        identity_path = run_dir / "identity.json"
        old = _read_record(identity_path)
        if old is not None and old != identity:
            raise ValueError(
                "initialization identity conflicts with the selected output directory"
            )
        _write_record(identity_path, identity)
        status_path = run_dir / "initialization.json"
        _write_record(status_path, {"status": "running", "fingerprint": fingerprint})
        outputs = {}
        try:
            for item in prepared:
                _assert_unchanged(item)
                directory = run_dir / item.spec["name"]
                directory.mkdir(parents=True, exist_ok=True)
                outputs[item.spec["name"]] = _initialize_dataset(
                    item, experts, directory, fingerprint=fingerprint, **settings
                )
                _assert_unchanged(item)
            result = {
                "status": "complete",
                "fingerprint": fingerprint,
                "output_dir": str(run_dir),
                "model_store": str(store),
                "datasets": outputs,
                "quality_passed": all(
                    info["quality_passed"] for info in outputs.values()
                ),
                "quality_policy": quality_policy,
                "quality_scope": "frozen_foundational_anchorer_only",
            }
            _write_record(status_path, result)
            return result
        except Exception as exc:
            _write_record(
                status_path,
                {
                    "status": "failed",
                    "fingerprint": fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise


def _preflight_datasets(
    specs,
    validation_fraction,
    max_windows,
    *,
    tuning_windows=12,
    confirmation_windows=12,
    min_confirmation_windows=4,
    quality_policy="required",
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
                    raw.get(key), f"{name}.{key}", minimum=2 if key == "lookback" else 1
                )
                spec[key] = raw[key]
            stride = spec["horizon"] if raw.get("stride") is None else raw["stride"]
            _positive_integer(stride, f"{name}.stride")
            spec["stride"] = stride
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
            total = len(dataset.frame)
            split = math.floor(total * (1 - validation_fraction))
            confirmation_start = split + (total - split) // 2
            if split < spec["lookback"] + spec["horizon"]:
                raise ValueError("training prefix is too short for lookback + horizon")
            selection_limit = (
                min(tuning_windows, max_windows) if max_windows else tuning_windows
            )
            confirmation_limit = (
                min(confirmation_windows, max_windows)
                if max_windows
                else confirmation_windows
            )
            origins = representative_origins(
                split, confirmation_start, spec["horizon"], stride, selection_limit
            )
            confirmation = representative_origins(
                confirmation_start, total, spec["horizon"], stride, confirmation_limit
            )
            if not origins or not confirmation:
                raise ValueError(
                    "selection and confirmation intervals each need a complete nonoverlapping target window"
                )
            if (
                quality_policy == "required"
                and len(confirmation) < min_confirmation_windows
            ):
                raise ValueError(
                    f"required quality needs at least {min_confirmation_windows} confirmation windows; available {len(confirmation)}"
                )
            prepared.append(
                _PreparedDataset(
                    spec,
                    dataset,
                    before,
                    split,
                    confirmation_start,
                    origins,
                    confirmation,
                )
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(
                f"{raw.get('name', '<unnamed>') if isinstance(raw, dict) else '<invalid>'}: {exc}"
            )
    if errors:
        raise ValueError("dataset preflight failed:\n- " + "\n- ".join(errors))
    return prepared


def _score_origin(item, origin, identity, path, predictor):
    record = _read_record(path)
    if record is None:
        history = (
            item.dataset.frame[item.spec["target_col"]]
            .iloc[origin - item.spec["lookback"] : origin]
            .to_numpy(dtype=float, copy=True)
        )
        timestamps = None
        if item.spec["timestamp_col"]:
            timestamps = (
                item.dataset.frame[item.spec["timestamp_col"]]
                .iloc[origin - item.spec["lookback"] : origin]
                .to_numpy(copy=True)
            )
        try:
            prediction = predictor(history, timestamps)
            target = (
                item.dataset.frame[item.spec["target_col"]]
                .iloc[origin : origin + item.spec["horizon"]]
                .to_numpy(dtype=float)
            )
            record = {
                "identity": identity,
                "origin": origin,
                "status": "valid",
                "mse": mse(prediction, target),
                "mae": mae(prediction, target),
                "error": "",
            }
        except Exception as exc:
            record = {
                "identity": identity,
                "origin": origin,
                "status": "failed",
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
        if any(
            isinstance(record.get(key), bool)
            or not isinstance(record.get(key), (int, float))
            or not math.isfinite(record[key])
            or record[key] < 0
            for key in ("mse", "mae")
        ):
            raise ValueError(f"scoring checkpoint contains invalid errors: {path}")
    elif (
        record.get("status") != "failed"
        or not isinstance(record.get("error"), str)
        or not record["error"]
    ):
        raise ValueError(f"scoring checkpoint has an invalid failure record: {path}")
    return record


def _score_recipe(item, candidate, origins, directory, fingerprint):
    candidate_id = content_fingerprint(candidate)
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
        "candidate_id": candidate_id,
        "origins": origins,
        "records": records,
        **summarize_metrics(records),
    }
    old = _read_record(directory / "trial.json")
    if old is not None and old != result:
        raise ValueError("candidate summary differs from its durable origin records")
    _write_record(directory / "trial.json", result)
    return result


def _initialize_dataset(
    item,
    experts,
    directory,
    *,
    fingerprint,
    tuning_rounds,
    seed,
    max_windows,
    tuning_windows,
    confirmation_windows,
    min_confirmation_windows,
    quality_policy,
    min_mse_improvement,
    max_mae_regression,
):
    spec, selected, tuning, defaults = item.spec, [], [], []
    for expert in experts:
        candidates = candidate_parameters(
            expert,
            lookback=spec["lookback"],
            seasonal_period=spec["seasonal_period"],
            tuning_rounds=tuning_rounds,
            seed=seed,
        )
        capacity = candidate_capacity(
            expert, lookback=spec["lookback"], seasonal_period=spec["seasonal_period"]
        )
        trials = []
        for number, params in enumerate(candidates):
            candidate = dict(expert, params=params)
            trial_dir = (
                directory
                / "selection"
                / expert["name"]
                / f"{number:03d}-{content_fingerprint(candidate)[:16]}"
            )
            trials.append(
                _score_recipe(item, candidate, item.origins, trial_dir, fingerprint)
            )
        valid = [
            (trial["mse"], trial["mae"], index, trial)
            for index, trial in enumerate(trials)
            if trial["status"] == "valid"
        ]
        summary = {
            "model": expert["name"],
            "requested_trials": tuning_rounds,
            "available_trials": capacity,
            "attempted_trials": len(trials),
            "successful_trials": len(valid),
            "failed_trials": len(trials) - len(valid),
            "pool_exhausted": len(trials) >= capacity,
            "validation_windows": len(item.origins),
        }
        if not valid:
            _write_record(
                directory / "selection_failed.json",
                {"model": expert["name"], "summary": summary, "trials": trials},
            )
            raise ValueError(
                f"{spec['name']}/{expert['name']}: no recipe succeeded on every selection origin; see {directory / 'selection_failed.json'}"
            )
        _, _, _, winner = min(valid, key=lambda row: row[:3])
        selected.append(winner["candidate"])
        summary.update(
            selected=winner["candidate"]["params"],
            mse=winner["mse"],
            mae=winner["mae"],
            baseline_valid=trials[0]["status"] == "valid",
        )
        tuning.append(summary)
        if trials[0]["status"] == "valid":
            defaults.append(trials[0])
    if not defaults:
        _write_record(
            directory / "selection_failed.json",
            {"error": "all default recipes failed", "selection": tuning},
        )
        raise ValueError(
            "no default expert succeeded on the complete selection design; no valid incumbent comparison"
        )
    incumbent = min(
        enumerate(defaults), key=lambda row: (row[1]["mse"], row[1]["mae"], row[0])
    )[1]
    timestamps = (
        item.dataset.frame[spec["timestamp_col"]] if spec["timestamp_col"] else None
    )
    protocol = {
        "split": "training_csv_only",
        "source_sha256": item.fingerprint,
        "source_rows": len(item.dataset.frame),
        "chronology": "strict_timestamps"
        if timestamps is not None
        else "supplied_row_order",
        "validation_start_row": item.split,
        "selection_start_row": item.split,
        "confirmation_start_row": item.confirmation_start,
        "library_end_row_exclusive": item.confirmation_start,
        "validation_origins": item.origins,
        "selection_origins": item.origins,
        "confirmation_origins": item.confirmation_origins,
        "target_windows_overlap": False,
        "tuning": "selection_only_rolling_origin_history_inputs",
        "final_library_scope": "frozen_source_prefix_excluding_confirmation_targets",
        "library_window_limit": max_windows,
        "historical_target_cutoff": str(timestamps.iloc[item.confirmation_start - 1])
        if timestamps is not None
        else item.confirmation_start - 1,
        "independent_test_evaluation": False,
        "default_incumbent_selected_on": "selection_MSE_then_MAE",
        "default_incumbent": incumbent["candidate"],
        "quality_scope": "frozen_foundational_anchorer_only",
    }
    manifest = {"experts": selected, "selection": tuning, "protocol": protocol}
    _write_record(directory / "selection.json", manifest)
    manifest_path = directory / "expert_manifest.json"
    atomic_write_json(manifest_path, manifest)
    library_path = directory / "anchor_library.json"
    completed = _read_record(directory / "library_complete.json")
    if completed is not None:
        if completed.get("pipeline_fingerprint") != fingerprint or completed.get(
            "manifest_sha256"
        ) != file_sha256(manifest_path):
            raise ValueError("completed library configuration changed")
        if not library_path.is_file() or completed.get("library_sha256") != file_sha256(
            library_path
        ):
            raise ValueError("completed anchor library is missing or changed")
    else:
        _assert_unchanged(item)
        library = build_anchor_library(
            spec["data_path"],
            lookback=spec["lookback"],
            horizon=spec["horizon"],
            seasonal_period=spec["seasonal_period"],
            dataset_name=spec["name"],
            target_col=spec["target_col"],
            timestamp_col=spec["timestamp_col"],
            stride=spec["stride"],
            max_windows=max_windows,
            expert_manifest=manifest_path,
            end_row=item.confirmation_start,
            checkpoint_dir=directory / "case_checkpoints",
        )
        _assert_unchanged(item)
        payload = library.to_dict()
        payload["initialization"] = {"pipeline_fingerprint": fingerprint, **protocol}
        atomic_write_json(library_path, payload)
        completed = {
            "pipeline_fingerprint": fingerprint,
            "manifest_sha256": file_sha256(manifest_path),
            "library_sha256": file_sha256(library_path),
            "case_count": len(library.cases),
        }
        _write_record(directory / "library_complete.json", completed)
    library = AnchorLibrary.load(library_path)
    if library.expert_specs != selected:
        raise ValueError("completed library does not use the selected expert recipes")
    report_path = directory / "quality_report.json"
    quality_complete = _read_record(directory / "quality_complete.json")
    if quality_complete is not None:
        if (
            quality_complete.get("library_sha256") != file_sha256(library_path)
            or not report_path.is_file()
            or quality_complete.get("report_sha256") != file_sha256(report_path)
        ):
            raise ValueError(
                "completed quality report or its frozen library is missing or changed"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _confirm_library(
            item,
            library,
            incumbent["candidate"],
            directory / "confirmation",
            fingerprint,
        )
        decision = assess_quality(
            report["candidate"],
            report["seasonal_naive"],
            report["default_incumbent"],
            minimum_windows=min_confirmation_windows,
            min_mse_improvement=min_mse_improvement,
            max_mae_regression=max_mae_regression,
        )
        report.update(
            decision,
            protocol=protocol,
            selection=tuning,
            quality_policy=quality_policy,
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
    if quality_policy == "required" and not report["quality_passed"]:
        raise ValueError(
            f"{spec['name']}: frozen Anchorer failed required confirmation quality ({', '.join(report['failed_checks'])}); see {report_path}"
        )
    return {
        "anchor_library": str(library_path),
        "expert_manifest": str(manifest_path),
        "case_count": len(library.cases),
        "cluster_count": len(library.clusters),
        "selection": tuning,
        "protocol": protocol,
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
                "origin": origin,
                "role": label,
                "recipe": incumbent
                if label == "default_incumbent"
                else seasonal
                if label == "seasonal_naive"
                else "frozen_saved_library",
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
    from backends import catalog, experts
    from anchorer import build, quality
    from utils import io, metrics

    modules = {
        "initialization": __file__,
        "catalog": catalog.__file__,
        "experts": experts.__file__,
        "library": build.__file__,
        "quality": quality.__file__,
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
