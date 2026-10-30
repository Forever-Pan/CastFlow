"""Training-side construction of reusable, cross-domain Experience Memory."""

from __future__ import annotations

import copy
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from anchorer import AnchorLibrary
from backends.catalog import file_sha256
from evaluation.parsing import aligned_arrays, parse_forecast_value
from forecasting.modules import PlanningModule, reflection_verdict
from utils.config import CastFlowConfig
from utils.io import (
    atomic_write_json,
    attach_covariate_forecasts,
    load_dataset,
)
from utils.metrics import mae, mse
from workflow.engine import CastFlow, request_identity, bounded_memory_windows
from workflow.memory import StrategyMemory
from workflow.recovery import ExecutionJournal, digest


def preflight_experience_inputs(
    dataset_specs: Sequence[dict[str, Any]],
    config: CastFlowConfig,
    *,
    max_windows: int | None = None,
    window_bounds: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Validate all training/covariate inputs without models, writes, or calls."""
    config.validate()
    if not config.api_ready():
        raise ValueError(
            "Experience Memory initialization requires the configured frozen teacher API"
        )
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows < 1
    ):
        raise ValueError("max_windows must be a positive integer or None")
    if (
        isinstance(dataset_specs, (str, bytes, dict))
        or not isinstance(dataset_specs, Sequence)
        or not dataset_specs
    ):
        raise ValueError(
            "dataset_specs must be a nonempty sequence of training dataset objects"
        )
    if window_bounds is not None and not isinstance(window_bounds, dict):
        raise ValueError(
            "window_bounds must map dataset names to target row boundaries"
        )
    if window_bounds is not None:
        names = {
            raw.get("name")
            for raw in dataset_specs
            if isinstance(raw, dict) and isinstance(raw.get("name"), str)
        }
        if set(window_bounds) != names:
            raise ValueError(
                "window_bounds must cover exactly the specified training datasets"
            )
    checked, seen = [], set()
    for raw in dataset_specs:
        if not isinstance(raw, dict):
            raise ValueError("dataset specification must be an object")
        name = raw.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in {".", ".."}
            or name.casefold() in seen
        ):
            raise ValueError("dataset names must be safe, nonempty, and unique")
        seen.add(name.casefold())
        if raw.get("split", "train") not in {"train", "train_val", "training"}:
            raise ValueError("Experience Memory accepts training data only")
        for key in ("lookback", "horizon", "seasonal_period"):
            value = raw.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (2 if key == "lookback" else 1)
            ):
                raise ValueError(
                    f"{name}.{key} must be a positive integer with lookback >= 2"
                )
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
                "Experience Memory requires an explicitly designated training CSV"
            )
        source_hash = file_sha256(source)
        dataset = load_dataset(source, raw.get("target_col"), raw.get("timestamp_col"))
        bounds = window_bounds[name] if window_bounds is not None else {}
        if not isinstance(bounds, dict) or (
            window_bounds is not None and set(bounds) != {"start_origin_row", "end_row"}
        ):
            raise ValueError(
                "memory window bounds require exactly start_origin_row and end_row"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in bounds.values()
        ):
            raise ValueError(
                "explicit memory window bounds must be integer row boundaries"
            )
        source_windows = bounded_memory_windows(
            dataset,
            raw["lookback"],
            raw["horizon"],
            stride=raw.get("memory_stride", raw.get("stride")),
            max_windows=max_windows,
            **bounds,
        )
        if not source_windows:
            raise ValueError(
                f"{name}: no complete training windows for memory exploration"
            )
        source_origins, windows = zip(*source_windows)
        covariate_path = raw.get("covariate_forecasts_path")
        covariate_hash = None
        if not isinstance(
            raw.get("covariate_issue_col", "issued_at"), str
        ) or not raw.get("covariate_issue_col", "issued_at"):
            raise ValueError("covariate_issue_col must be a nonempty column name")
        if covariate_path:
            covariate_path = Path(covariate_path).expanduser().resolve(strict=True)
            covariate_hash = file_sha256(covariate_path)
            covariates = pd.read_csv(covariate_path, keep_default_na=False)
            for window in windows:
                attach_covariate_forecasts(
                    window,
                    covariates,
                    raw["horizon"],
                    issued_at=raw.get("covariate_issued_at"),
                    issue_col=raw.get("covariate_issue_col", "issued_at"),
                )
            if file_sha256(covariate_path) != covariate_hash:
                raise ValueError(
                    f"{name}: covariate forecasts changed during preflight"
                )
        elif raw.get("covariate_issued_at") is not None:
            raise ValueError("covariate_issued_at requires covariate_forecasts_path")
        if file_sha256(source) != source_hash:
            raise ValueError(f"{name}: training CSV changed during preflight")
        checked.append(
            {
                "name": name,
                "source_sha256": source_hash,
                "windows": len(windows),
                "source_origins": list(source_origins),
                "window_bounds": bounds,
                "covariate_forecasts_sha256": covariate_hash,
            }
        )
    return {"datasets": checked, "teacher_configured": True}


def initialize_experience_memories(
    dataset_specs: Sequence[dict[str, Any]],
    case_initialization: dict[str, Any],
    config: CastFlowConfig,
    *,
    resume: bool = True,
    max_windows: int | None = None,
    show_progress: bool = False,
    verbose_samples: bool = False,
) -> dict[str, Any]:
    """Build accepted training trajectories and a combined retrieval artifact.

    Exploration uses the frozen teacher for planning, numerical responses, and
    reflection. It does not train or invoke the inference-side specialized model.
    Completed dataset artifacts are checksum-verified and reused as a whole,
    including when some training windows produced no accepted trajectory.
    """
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows < 1
    ):
        raise ValueError("max_windows must be a positive integer or None")
    if (
        not isinstance(case_initialization, dict)
        or case_initialization.get("status") != "complete"
    ):
        raise ValueError(
            "case-library initialization must complete before constructing Experience Memory"
        )
    bounds = {
        name: {
            "start_origin_row": info.get("protocol", {}).get(
                "memory_start_origin_row", specs["lookback"]
            ),
            "end_row": info.get("protocol", {}).get(
                "memory_end_row_exclusive",
                info.get("protocol", {}).get("confirmation_start_row"),
            ),
        }
        for specs in dataset_specs
        for name in [specs["name"]]
        for info in [case_initialization.get("datasets", {}).get(name, {})]
    }
    preflight_experience_inputs(
        dataset_specs, config, max_windows=max_windows, window_bounds=bounds
    )
    jobs = _preflight(dataset_specs, case_initialization, config, max_windows)
    identity = {
        "schema_version": 1,
        "case_initialization_fingerprint": case_initialization.get("fingerprint"),
        "datasets": [job["identity"] for job in jobs],
        "max_windows": max_windows,
    }
    fingerprint = digest(identity)
    suffix = (
        fingerprint[:20] if resume else f"{fingerprint[:20]}-{uuid.uuid4().hex[:8]}"
    )
    output = (
        Path(case_initialization["output_dir"]).expanduser().resolve()
        / "experience_memory"
        / suffix
    )
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "memory_initialization.json"
    with ExecutionJournal(
        output / "stages", "experience-memory:" + fingerprint, identity
    ) as journal:
        atomic_write_json(
            status_path, {"status": "running", "fingerprint": fingerprint}
        )
        try:
            records = []
            for job in jobs:
                _assert_inputs(job)
                record = journal.run(
                    "dataset:" + job["name"],
                    job["identity"],
                    lambda job=job: _build_dataset(
                        job, output, max_windows, show_progress, verbose_samples
                    ),
                )
                _validate_artifact(record, job)
                _assert_inputs(job)
                records.append(record)
            combined = journal.run(
                "combined_memory",
                records,
                lambda: _combine(records, output / "experience_memory.json"),
            )
            if (
                not Path(combined["experience_memory"]).is_file()
                or file_sha256(combined["experience_memory"]) != combined["sha256"]
            ):
                raise ValueError("combined Experience Memory is missing or changed")
            if (
                len(StrategyMemory.load(combined["experience_memory"]).entries)
                != combined["entries"]
            ):
                raise ValueError("combined Experience Memory entry count changed")
            result = {
                "status": "complete",
                "fingerprint": fingerprint,
                "output_dir": str(output),
                **combined,
                "datasets": {record["dataset_name"]: record for record in records},
                "scope": "training_trajectories_only",
                "inference_specialist_configured": config.local_forecast_ready(),
            }
            atomic_write_json(status_path, result)
            return result
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for secret in (config.api_key, config.local_model_api_key):
                if secret and secret != "EMPTY":
                    message = message.replace(secret, "[redacted]")
            atomic_write_json(
                status_path,
                {"status": "failed", "fingerprint": fingerprint, "error": message},
            )
            raise
        finally:
            for job in jobs:
                job["workflow"].llm.close()
                job["workflow"].local_llm.close()


def _preflight(specs, case_initialization, config, max_windows):
    if (
        isinstance(specs, (str, bytes, dict))
        or not isinstance(specs, Sequence)
        or not specs
    ):
        raise ValueError(
            "dataset_specs must be a nonempty sequence of training dataset objects"
        )
    cases = case_initialization.get("datasets")
    if not isinstance(cases, dict):
        raise ValueError("completed case initialization has no dataset artifacts")
    jobs, seen = [], set()
    for raw in specs:
        if not isinstance(raw, dict):
            raise ValueError("dataset specification must be an object")
        name = raw.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in {".", ".."}
        ):
            raise ValueError("dataset name must be a safe nonempty identifier")
        if name.casefold() in seen:
            raise ValueError("dataset names must be unique")
        seen.add(name.casefold())
        if raw.get("split", "train") not in {"train", "train_val", "training"}:
            raise ValueError("Experience Memory accepts training data only")
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
                "Experience Memory requires an explicitly designated training CSV"
            )
        info = cases.get(name)
        if not isinstance(info, dict):
            raise ValueError(
                f"no completed anchor library for training dataset {name!r}"
            )
        inference_library_path = (
            Path(info["anchor_library"]).expanduser().resolve(strict=True)
        )
        inference_library_hash = file_sha256(inference_library_path)
        if (
            info.get("anchor_library_sha256", inference_library_hash)
            != inference_library_hash
        ):
            raise ValueError(f"{name}: inference library changed")
        library_path = (
            Path(info.get("memory_anchor_library", info["anchor_library"]))
            .expanduser()
            .resolve(strict=True)
        )
        library_hash = file_sha256(library_path)
        if info.get("memory_anchor_library_sha256", library_hash) != library_hash:
            raise ValueError(f"{name}: prepared training library changed")
        library = AnchorLibrary.load(library_path)
        for key in ("lookback", "horizon", "seasonal_period"):
            if isinstance(raw.get(key), bool) or raw.get(key) != getattr(library, key):
                raise ValueError(
                    f"{name}: {key} differs from the completed anchor library"
                )
        if library.dataset_name != name:
            raise ValueError(f"{name}: anchor library belongs to a different dataset")
        target = raw.get("target_col") or library.target_col
        timestamp = raw.get("timestamp_col") or library.timestamp_col
        if target != library.target_col or timestamp != library.timestamp_col:
            raise ValueError(
                f"{name}: target/timestamp columns differ from the anchor library"
            )
        source_hash = file_sha256(source)
        if info.get("protocol", {}).get("source_sha256") != source_hash:
            raise ValueError(
                f"{name}: training CSV differs from the case-library source"
            )
        dataset = load_dataset(source, target, timestamp)
        if file_sha256(source) != source_hash:
            raise ValueError(f"{name}: training data changed during preflight")
        stride = raw.get("memory_stride", raw.get("stride"))
        protocol = info.get("protocol", {})
        start_origin_row = protocol.get("memory_start_origin_row", library.lookback)
        end_row = protocol.get(
            "memory_end_row_exclusive", protocol.get("confirmation_start_row")
        )
        source_windows = bounded_memory_windows(
            dataset,
            library.lookback,
            library.horizon,
            stride=stride,
            max_windows=max_windows,
            start_origin_row=start_origin_row,
            end_row=end_row,
        )
        if not source_windows:
            raise ValueError(
                f"{name}: no complete training windows for memory exploration"
            )
        source_origins, windows = zip(*source_windows)
        covariate_path = raw.get("covariate_forecasts_path")
        covariate_hash = None
        if covariate_path:
            covariate_path = Path(covariate_path).expanduser().resolve(strict=True)
            covariate_hash = file_sha256(covariate_path)
            covariates = pd.read_csv(covariate_path, keep_default_na=False)
            for window in windows:
                attach_covariate_forecasts(
                    window,
                    covariates,
                    library.horizon,
                    issued_at=raw.get("covariate_issued_at"),
                    issue_col=raw.get("covariate_issue_col", "issued_at"),
                )
            if file_sha256(covariate_path) != covariate_hash:
                raise ValueError(
                    f"{name}: covariate forecasts changed during preflight"
                )
        runtime = replace(
            config,
            lookback=library.lookback,
            horizon=library.horizon,
            seasonal_period=library.seasonal_period,
            target_col=target,
            timestamp_col=timestamp,
            dataset_name=name,
            anchor_library_path=str(library_path),
            allow_train_only_tools=True,
        )
        runtime.validate()
        workflow = CastFlow(runtime)
        runtime_identity = request_identity(windows[0], runtime, workflow.forecasting)
        identity = {
            "name": name,
            "source_path": str(source),
            "source_sha256": source_hash,
            "anchor_library": str(library_path),
            "anchor_library_sha256": library_hash,
            "inference_anchor_library": str(inference_library_path),
            "inference_anchor_library_sha256": inference_library_hash,
            "covariate_forecasts_path": str(covariate_path) if covariate_path else None,
            "covariate_forecasts_sha256": covariate_hash,
            "covariate_issued_at": raw.get("covariate_issued_at"),
            "covariate_issue_col": raw.get("covariate_issue_col", "issued_at"),
            "stride": stride,
            "start_origin_row": start_origin_row,
            "end_row": end_row,
            "source_origins": list(source_origins),
            "window_count": len(windows),
            "runtime": runtime_identity,
        }
        jobs.append(
            {
                "name": name,
                "source": source,
                "source_hash": source_hash,
                "library_path": library_path,
                "library_hash": library_hash,
                "inference_library_path": inference_library_path,
                "inference_library_hash": inference_library_hash,
                "covariate_path": covariate_path,
                "covariate_hash": covariate_hash,
                "covariate_issued_at": raw.get("covariate_issued_at"),
                "covariate_issue_col": raw.get("covariate_issue_col", "issued_at"),
                "stride": stride,
                "start_origin_row": start_origin_row,
                "end_row": end_row,
                "workflow": workflow,
                "window": windows[0],
                "identity": identity,
            }
        )
    if set(cases) != {job["name"] for job in jobs}:
        raise ValueError(
            "training specifications must cover exactly the completed case datasets"
        )
    return jobs


def _assert_inputs(job):
    if (
        file_sha256(job["source"]) != job["source_hash"]
        or file_sha256(job["library_path"]) != job["library_hash"]
        or file_sha256(job["inference_library_path"]) != job["inference_library_hash"]
    ):
        raise ValueError(f"{job['name']}: training source or anchor library changed")
    if (
        job["covariate_path"]
        and file_sha256(job["covariate_path"]) != job["covariate_hash"]
    ):
        raise ValueError(f"{job['name']}: supplied covariate forecasts changed")
    workflow = job["workflow"]
    if (
        request_identity(job["window"], workflow.config, workflow.forecasting)
        != job["identity"]["runtime"]
    ):
        raise ValueError(
            "memory initialization implementation or configuration changed during execution"
        )


def _build_dataset(job, output, max_windows, show_progress, verbose_samples):
    path = output / "datasets" / job["name"] / "experience_memory.json"
    memory = job["workflow"].build_memory_from_csv(
        job["source"],
        path,
        stride=job["stride"],
        max_windows=max_windows,
        show_progress=show_progress,
        resume=True,
        verbose_samples=verbose_samples,
        covariate_forecasts_path=job["covariate_path"],
        covariate_issued_at=job["covariate_issued_at"],
        covariate_issue_col=job["covariate_issue_col"],
        completion_journal_dir=path.parent / "completed_windows",
        start_origin_row=job["start_origin_row"],
        end_row=job["end_row"],
    )
    _validate_entries(memory, job)
    return {
        "dataset_name": job["name"],
        "experience_memory": str(path),
        "sha256": file_sha256(path),
        "entries": len(memory.entries),
        "training_windows": job["identity"]["window_count"],
        "unretained_windows": job["identity"]["window_count"] - len(memory.entries),
        "anchor_library": str(job["inference_library_path"]),
        "anchor_library_sha256": job["inference_library_hash"],
        "memory_anchor_library": str(job["library_path"]),
        "memory_anchor_library_sha256": job["library_hash"],
        "source_sha256": job["source_hash"],
        "start_origin_row": job["start_origin_row"],
        "end_row": job["end_row"],
    }


def _validate_entries(memory, job):
    if not memory.entries:
        raise ValueError(
            f"{job['name']}: no format-and-reflection-accepted training trajectory; memory is not ready"
        )
    seen = set()
    for entry in memory.entries:
        if (
            entry.get("dataset_name") != job["name"]
            or entry.get("source_fingerprint") != job["source_hash"]
        ):
            raise ValueError("Experience Memory contains a different dataset or source")
        if entry.get("idx") in seen:
            raise ValueError(
                "Experience Memory contains duplicate training-window identifiers"
            )
        seen.add(entry.get("idx"))
        index = entry.get("idx")
        origins = job["identity"]["source_origins"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(origins)
            or entry.get("source_origin_row") != origins[index]
        ):
            raise ValueError(
                "Experience Memory target origin differs from the bounded training design"
            )
        if not set(PlanningModule.mandatory_tools).issubset(
            entry.get("tool_schedule", [])
        ):
            raise ValueError(
                "accepted memory trajectory is missing mandatory tool evidence"
            )
        if reflection_verdict(entry.get("trajectory", "")) is not True:
            raise ValueError(
                "Experience Memory contains a trajectory without semantic acceptance"
            )
        prediction = parse_forecast_value(entry.get("Forecasting_answer", ""))
        target = parse_forecast_value(entry.get("ground_truth", ""))
        predicted, actual = aligned_arrays(prediction, target)
        if len(predicted) != job["workflow"].config.horizon:
            raise ValueError(
                "Experience Memory trajectory has an incorrect forecast horizon"
            )
        metrics = entry.get("metrics", {})
        for name, actual_value in (
            ("mse", mse(predicted, actual)),
            ("mae", mae(predicted, actual)),
        ):
            saved = metrics.get(name)
            if (
                isinstance(saved, bool)
                or not isinstance(saved, (int, float))
                or not math.isfinite(saved)
                or not math.isclose(saved, actual_value, rel_tol=1e-10, abs_tol=1e-10)
            ):
                raise ValueError(
                    "Experience Memory metrics do not match the accepted training response"
                )


def _validate_artifact(record, job):
    path = Path(record["experience_memory"])
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("completed Experience Memory is missing or changed")
    memory = StrategyMemory.load(path)
    if len(memory.entries) != record.get("entries"):
        raise ValueError("completed Experience Memory entry count changed")
    _validate_entries(memory, job)


def _combine(records, path):
    combined = StrategyMemory()
    for record in records:
        combined.entries.extend(
            copy.deepcopy(StrategyMemory.load(record["experience_memory"]).entries)
        )
    combined.save(path)
    return {
        "experience_memory": str(path),
        "sha256": file_sha256(path),
        "entries": len(combined.entries),
    }
