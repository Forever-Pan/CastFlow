"""Resumable training-side preparation, expert selection, and anchor-library build."""

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

from anchorer.build import AnchorLibrary, build_anchor_library
from backends.catalog import (
    candidate_parameters,
    content_fingerprint,
    file_sha256,
    preflight_models,
    prepare_models,
    selected_model_names,
)
from backends.experts import forecast_expert_specs
from utils.io import DatasetView, atomic_write_json, load_dataset
from utils.metrics import mse


@dataclass(slots=True)
class _PreparedDataset:
    spec: dict[str, Any]
    dataset: DatasetView
    fingerprint: str
    split: int
    origins: list[int]


def initialize_case_libraries(
    dataset_specs: Sequence[dict[str, Any]],
    output_root: str | Path,
    model_store: str | Path,
    *,
    models: Sequence[str] | None = None,
    tuning_rounds: int = 3,
    validation_fraction: float = 0.2,
    seed: int = 0,
    download_models: bool = False,
    max_windows: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Prepare selected experts and build one library per supplied training CSV.

    The final fraction of each training CSV is used for chronological rolling-origin
    parameter selection. Every forecast is fitted only to its own lookback. After
    selection, the frozen recipes build the library from the complete training CSV.
    No test split is loaded, and these tuning scores are not independent test scores.
    """
    _positive_integer(tuning_rounds, "tuning_rounds")
    if tuning_rounds > 32:
        raise ValueError("tuning_rounds must not exceed the bounded limit of 32")
    _positive_integer(seed, "seed", minimum=0)
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError(
            "validation_fraction must be finite and strictly between zero and one"
        )
    if not isinstance(download_models, bool) or not isinstance(resume, bool):
        raise ValueError("download_models and resume must be boolean")
    if max_windows is not None:
        _positive_integer(max_windows, "max_windows")
    names = selected_model_names(models)
    # Complete all inexpensive data/model checks before creating output or loading weights.
    prepared = _preflight_datasets(dataset_specs, validation_fraction, max_windows)
    for item in prepared:
        if (
            any(name in {"Ridge", "XGBoost", "LightGBM"} for name in names)
            and item.spec["lookback"] < 3
        ):
            raise ValueError("autoregressive experts require lookback >= 3")
        if "AutoARIMA" in names and item.spec["lookback"] < 4:
            raise ValueError("AutoARIMA requires lookback >= 4")
    output = Path(output_root).expanduser().resolve()
    store = Path(model_store).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"output_root is not a directory: {output}")
    preflight_models(names, store, download_models=download_models)
    store.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(store / ".preparation.lock"):
        experts = prepare_models(names, store, download_models=download_models)
        model_metadata = preflight_models(names, store, download_models=False)
    identity = {
        "schema_version": 1,
        "datasets": [
            {
                "spec": item.spec,
                "sha256": item.fingerprint,
                "split_row": item.split,
                "validation_origins": item.origins,
            }
            for item in prepared
        ],
        "experts": experts,
        "model_metadata": model_metadata,
        "tuning_rounds": tuning_rounds,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "max_windows": max_windows,
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
        previous_identity = _read_record(identity_path)
        if previous_identity is not None and previous_identity != identity:
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
                dataset_dir = run_dir / item.spec["name"]
                dataset_dir.mkdir(parents=True, exist_ok=True)
                outputs[item.spec["name"]] = _initialize_dataset(
                    item,
                    experts,
                    dataset_dir,
                    tuning_rounds=tuning_rounds,
                    seed=seed,
                    max_windows=max_windows,
                    fingerprint=fingerprint,
                )
                _assert_unchanged(item)
            result = {
                "status": "complete",
                "fingerprint": fingerprint,
                "output_dir": str(run_dir),
                "model_store": str(store),
                "datasets": outputs,
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
    specs: Sequence[dict[str, Any]], validation_fraction: float, max_windows: int | None
) -> list[_PreparedDataset]:
    if (
        isinstance(specs, (str, bytes, dict))
        or not isinstance(specs, Sequence)
        or not specs
    ):
        raise ValueError(
            "dataset_specs must be a nonempty sequence of training dataset objects"
        )
    prepared = []
    errors = []
    names: set[str] = set()
    for raw in specs:
        try:
            if not isinstance(raw, dict):
                raise ValueError("dataset specification must be an object")
            name = raw.get("name")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or name in {".", ".."}
            ):
                raise ValueError("dataset name must be a safe, nonempty identifier")
            if name.casefold() in names:
                raise ValueError(f"duplicate dataset name: {name}")
            names.add(name.casefold())
            if raw.get("split", "train") not in {"train", "train_val", "training"}:
                raise ValueError(f"{name}: initialization accepts training data only")
            source = Path(raw["data_path"]).expanduser().resolve(strict=True)
            if not source.is_file() or source.suffix.casefold() != ".csv":
                raise ValueError(f"{name}: data_path must be a CSV file")
            if any(
                re.search(r"(^|[_\-.])test($|[_\-.])", part.casefold())
                for part in source.parts
            ):
                raise ValueError(
                    f"{name}: test-named input paths cannot be used for training initialization"
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
                raise ValueError(f"{name}: data changed during preflight")
            spec["target_col"], spec["timestamp_col"] = (
                dataset.target_col,
                dataset.timestamp_col,
            )
            split = int(math.floor(len(dataset.frame) * (1 - validation_fraction)))
            if split < spec["lookback"] + spec["horizon"]:
                raise ValueError(
                    f"{name}: training prefix is too short for lookback + horizon"
                )
            if len(dataset.frame) - split < spec["horizon"]:
                raise ValueError(
                    f"{name}: chronological validation tail is shorter than the horizon"
                )
            origins = list(
                range(split, len(dataset.frame) - spec["horizon"] + 1, stride)
            )
            if max_windows is not None:
                origins = origins[:max_windows]
            if not origins:
                raise ValueError(f"{name}: no chronological validation windows")
            prepared.append(_PreparedDataset(spec, dataset, before, split, origins))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(
                f"{raw.get('name', '<unnamed>') if isinstance(raw, dict) else '<invalid>'}: {exc}"
            )
    if errors:
        raise ValueError("dataset preflight failed:\n- " + "\n- ".join(errors))
    return prepared


def _initialize_dataset(
    item: _PreparedDataset,
    experts: list[dict[str, Any]],
    directory: Path,
    *,
    tuning_rounds: int,
    seed: int,
    max_windows: int | None,
    fingerprint: str,
) -> dict[str, Any]:
    spec = item.spec
    values = item.dataset.frame[spec["target_col"]].to_numpy(dtype=float)
    selected = []
    tuning = []
    for expert in experts:
        candidates = candidate_parameters(
            expert,
            lookback=spec["lookback"],
            seasonal_period=spec["seasonal_period"],
            tuning_rounds=tuning_rounds,
            seed=seed,
        )
        scores = []
        for number, params in enumerate(candidates):
            candidate = dict(expert, params=params)
            candidate_id = content_fingerprint(candidate)
            path = (
                directory
                / "selection"
                / expert["name"]
                / f"{number:02d}-{candidate_id[:16]}.json"
            )
            completed = _read_record(path)
            if completed is None:
                errors = []
                for origin in item.origins:
                    history = values[origin - spec["lookback"] : origin].copy()
                    # The predictor receives history only; future labels enter only this scorer.
                    timestamps = None
                    if spec["timestamp_col"]:
                        timestamps = (
                            item.dataset.frame[spec["timestamp_col"]]
                            .iloc[origin - spec["lookback"] : origin]
                            .to_numpy(copy=True)
                        )
                    predictions, _ = forecast_expert_specs(
                        [candidate],
                        history,
                        spec["horizon"],
                        spec["seasonal_period"],
                        timestamps=timestamps,
                    )
                    errors.append(
                        mse(
                            predictions[expert["name"]],
                            values[origin : origin + spec["horizon"]],
                        )
                    )
                score = float(np.mean(errors))
                if not math.isfinite(score):
                    raise ValueError(
                        f"{spec['name']}/{expert['name']}: validation MSE is nonfinite"
                    )
                completed = {
                    "candidate": candidate,
                    "mse": score,
                    "origins": item.origins,
                    "windows": len(errors),
                    "pipeline_fingerprint": fingerprint,
                }
                _write_record(path, completed)
            if (
                completed.get("candidate") != candidate
                or completed.get("origins") != item.origins
                or completed.get("pipeline_fingerprint") != fingerprint
            ):
                raise ValueError(
                    f"cached parameter selection has incompatible identity: {path}"
                )
            score = completed.get("mse")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or score < 0
            ):
                raise ValueError(f"cached parameter selection has invalid MSE: {path}")
            scores.append((float(score), number, candidate))
        best_score, _, best = min(scores, key=lambda row: (row[0], row[1]))
        selected.append(best)
        tuning.append(
            {
                "model": expert["name"],
                "selected": best["params"],
                "mse": best_score,
                "candidate_count": len(candidates),
                "validation_windows": len(item.origins),
            }
        )
    timestamps = (
        item.dataset.frame[spec["timestamp_col"]] if spec["timestamp_col"] else None
    )
    protocol = {
        "split": "training_csv_only",
        "source_sha256": item.fingerprint,
        "source_rows": len(values),
        "chronology": (
            "strict_timestamps" if timestamps is not None else "supplied_row_order"
        ),
        "validation_start_row": item.split,
        "validation_origins": item.origins,
        "tuning": "rolling_origin_history_only",
        "final_library_scope": "complete_training_csv_after_parameter_selection",
        "library_window_limit": max_windows,
        "historical_target_cutoff": (
            str(timestamps.iloc[-1]) if timestamps is not None else len(values) - 1
        ),
        "independent_test_evaluation": False,
    }
    manifest_path = directory / "expert_manifest.json"
    manifest = {"experts": selected, "selection": tuning, "protocol": protocol}
    _write_record(directory / "selection.json", manifest)
    atomic_write_json(manifest_path, manifest)
    complete_path = directory / "library_complete.json"
    library_path = directory / "anchor_library.json"
    completed = _read_record(complete_path)
    if completed is not None:
        if completed.get("pipeline_fingerprint") != fingerprint or completed.get(
            "manifest_sha256"
        ) != file_sha256(manifest_path):
            raise ValueError(
                "completed library configuration changed; use a distinct initialization output"
            )
        if not library_path.is_file() or completed.get("library_sha256") != file_sha256(
            library_path
        ):
            raise ValueError(
                "completed anchor library is missing or changed; restore it or start a fresh initialization"
            )
        library = AnchorLibrary.load(library_path)
        if library.expert_specs != selected:
            raise ValueError(
                "completed library does not use the selected expert parameters"
            )
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
        _write_record(complete_path, completed)
    return {
        "anchor_library": str(library_path),
        "expert_manifest": str(manifest_path),
        "case_count": len(library.cases),
        "cluster_count": len(library.clusters),
        "selection": tuning,
        "protocol": protocol,
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
    from anchorer import build

    modules = {
        "initialization": __file__,
        "catalog": catalog.__file__,
        "experts": experts.__file__,
        "library": build.__file__,
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
