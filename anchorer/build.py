from __future__ import annotations

import json
import hashlib
import warnings
import copy
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from utils.io import DatasetView, load_dataset, make_windows
from utils.schema import TimeSeriesWindow
from anchorer.models import forecast_with_anchorer_models
from utils.metrics import mse
from backends.experts import (
    finite_vector,
    forecast_expert_specs,
    load_expert_manifest,
    validate_expert_specs,
)


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
    history_start_row: int | None = None
    target_start_row: int | None = None
    target_end_row: int | None = None


@dataclass(slots=True)
class AnchorCluster:
    cluster_id: int
    window: list[float]
    best_model: dict[str, int]
    total_weight: int
    medoid_case_id: int
    size: int
    model_errors: dict[str, float] = field(default_factory=dict)
    case_ids: list[int] = field(default_factory=list)
    weighting_losses: dict[str, float] = field(default_factory=dict)
    calibration_count: int = 0
    reference_case_ids: list[int] = field(default_factory=list)


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
    expert_specs: list[dict[str, Any]] = field(default_factory=list)
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    coverage: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, self.to_dict())
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
            "weighting": "historical_loss_softmax",
            "distance_weights": list(self.distance_weights),
            "expert_specs": self.expert_specs,
            "coverage": self.coverage,
            "calibration": self.calibration,
            **({"promotion": copy.deepcopy(self.promotion)} if self.promotion else {}),
            "cases": [asdict(case) for case in self.cases],
            "clusters": [asdict(cluster) for cluster in self.clusters],
            "case_base": [
                {"window": case.window, "best_model": case.best_model}
                for case in self.cases
            ],
            "case_neighbor": [
                {
                    "look_back_window": case.target_history,
                    "pred_window": case.target_future,
                }
                for case in self.cases
            ],
            "cluster_base": [
                {
                    "window": cluster.window,
                    "best_model": cluster.best_model,
                    "total_weight": cluster.total_weight,
                    "model_errors": cluster.model_errors,
                    "case_ids": cluster.case_ids,
                    "weighting_losses": cluster.weighting_losses,
                    "calibration_count": cluster.calibration_count,
                    "reference_case_ids": cluster.reference_case_ids,
                }
                for cluster in self.clusters
            ],
        }

    def save_case_library(
        self, path: str | Path, data_path: str | Path | None = None
    ) -> Path:
        """Save the library in the old case-library split JSON layout."""
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        write_json(output / "anchor_library.json", payload)
        write_json(output / "case_base.json", payload["case_base"])
        write_json(output / "case_neighbor.json", payload["case_neighbor"])
        write_json(output / "cluster_base.json", payload["cluster_base"])
        write_json(
            output / "cases_stats.json",
            dict(Counter(case.best_model for case in self.cases)),
        )
        if data_path is not None:
            write_json(
                output / "memory.json",
                build_series_memory_stats(
                    data_path, self.target_col, self.timestamp_col
                ),
            )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AnchorLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchorLibrary":
        if not isinstance(payload, dict):
            raise ValueError("anchor library must be a JSON object")
        weights = normalize_distance_weights(
            payload.get("distance_weights", (1 / 3, 1 / 3, 1 / 3))
        )
        cases = [anchor_case_from_dict(case) for case in payload.get("cases", [])]
        clusters = [AnchorCluster(**cluster) for cluster in payload.get("clusters", [])]
        if not cases or not clusters:
            raise ValueError(
                "anchor library must contain historical cases and medoid clusters"
            )
        if any(not cluster.model_errors for cluster in clusters):
            if any(not case.model_errors for case in cases):
                raise ValueError(
                    "anchor library has no historical loss records; rebuild it before forecasting"
                )
            clusters = build_anchor_clusters(
                cases, len(clusters), distance_weights=weights
            )
            warnings.warn(
                "Rebuilt anchor clusters from stored historical losses; save this library in version 2 format.",
                UserWarning,
                stacklevel=2,
            )
        for cluster in clusters:
            validate_model_errors(cluster.model_errors)
            if cluster.weighting_losses:
                validate_model_errors(cluster.weighting_losses)
                if not set(cluster.weighting_losses).issubset(cluster.model_errors):
                    raise ValueError(
                        "calibrated active experts are absent from raw loss metadata"
                    )
            finite_vector(cluster.window, "cluster medoid")
        library = cls(
            version="3" if str(payload.get("version")) == "3" else "2",
            lookback=int(payload["lookback"]),
            horizon=int(payload["horizon"]),
            seasonal_period=int(payload["seasonal_period"]),
            target_col=str(payload["target_col"]),
            timestamp_col=payload.get("timestamp_col"),
            dataset_name=payload.get("dataset_name"),
            model_source=str(payload.get("model_source", "foundational_anchorer")),
            cases=cases,
            clusters=clusters,
            expert_specs=(
                validate_expert_specs(payload["expert_specs"])
                if payload.get("expert_specs")
                else []
            ),
            distance_weights=weights,
            coverage=copy.deepcopy(payload.get("coverage", {})),
            calibration=copy.deepcopy(payload.get("calibration", {})),
            promotion=copy.deepcopy(payload.get("promotion", {})),
        )
        if library.lookback < 2 or library.horizon < 1 or library.seasonal_period < 1:
            raise ValueError(
                "anchor library dimensions must be positive with lookback greater than one"
            )
        if any(
            len(case.target_history) != library.lookback
            or len(case.target_future) != library.horizon
            for case in cases
        ):
            raise ValueError("anchor case dimensions do not match library metadata")
        if any(len(case.window) != library.lookback for case in cases) or any(
            len(cluster.window) != library.lookback for cluster in clusters
        ):
            raise ValueError("anchor representations do not match the library lookback")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("anchor case IDs must be unique")
        if not isinstance(library.coverage, dict) or not isinstance(
            library.calibration, dict
        ):
            raise ValueError("anchor coverage and calibration metadata must be objects")
        if library.calibration and any(
            not cluster.weighting_losses for cluster in clusters
        ):
            raise ValueError("calibrated library is missing cluster weighting losses")
        if (
            any(cluster.weighting_losses for cluster in clusters)
            and not library.calibration
        ):
            raise ValueError(
                "calibrated weighting losses require their historical calibration metadata"
            )
        for case in cases:
            rows = (case.history_start_row, case.target_start_row, case.target_end_row)
            if any(value is not None for value in rows) and (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in rows
                )
                or case.target_start_row != case.history_start_row + library.lookback
                or case.target_end_row != case.target_start_row + library.horizon
            ):
                raise ValueError(
                    "anchor case row boundaries do not match its dimensions"
                )
        if library.calibration:
            families = expert_families(library)
            for cluster in clusters:
                active = [families[name] for name in cluster.weighting_losses]
                if len(active) != len(set(active)):
                    raise ValueError(
                        "calibrated clusters may activate only one variant per expert family"
                    )
        if not isinstance(library.promotion, dict):
            raise ValueError("anchor promotion metadata must be an object")
        if library.promotion:
            from anchorer.policies import validate_promotion_metadata

            validate_promotion_metadata(library)
        return library


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
    expert_manifest: str | Path | None = None,
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    end_row: int | None = None,
    start_row: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> AnchorLibrary:
    weights = normalize_distance_weights(distance_weights)
    expert_specs = load_expert_manifest(expert_manifest)
    source = load_dataset(data_path, target_col, timestamp_col)
    dataset, origins = _eligible_anchor_windows(
        source, lookback, horizon, stride, start_row, end_row, max_windows
    )
    _validate_expert_fit_cutoffs(expert_specs, origins[0] if origins else 0, data_path)
    windows = [_window_at(dataset, origin, lookback, horizon) for origin in origins]
    if not windows:
        raise ValueError(
            "no anchor cases could be built; check lookback/horizon against dataset length"
        )

    checkpoints, checkpoint_identity = _prepare_case_checkpoints(
        checkpoint_dir,
        dataset,
        lookback,
        horizon,
        seasonal_period,
        dataset_name,
        stride,
        max_windows,
        expert_specs,
        weights,
        origins,
    )
    cases = []
    for idx, window in enumerate(
        progress_iter(
            windows, label="Building anchor cases", unit="case", enabled=show_progress
        )
    ):
        path = checkpoints / f"case_{idx:08d}.json" if checkpoints is not None else None
        if path is not None and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record.get("payload") if isinstance(record, dict) else None
            if (
                not isinstance(payload, dict)
                or record.get("sha256") != _checkpoint_digest(payload)
                or payload.get("identity") != checkpoint_identity
            ):
                raise ValueError(
                    f"anchor case checkpoint is corrupt or incompatible: {path}"
                )
            case = anchor_case_from_dict(payload["case"])
            if (
                case.case_id != idx
                or case.target_history != window.target_history.astype(float).tolist()
                or case.target_future != window.target_future.astype(float).tolist()
            ):
                raise ValueError(
                    f"anchor case checkpoint does not match its source window: {path}"
                )
        else:
            case = build_anchor_case(
                idx,
                window,
                seasonal_period,
                dataset_name=dataset_name,
                quiet_warnings=quiet_warnings,
                expert_specs=expert_specs,
            )
            case.history_start_row = origins[idx] - lookback
            case.target_start_row = origins[idx]
            case.target_end_row = origins[idx] + horizon
            if path is not None:
                payload = {"identity": checkpoint_identity, "case": asdict(case)}
                write_json(
                    path, {"payload": payload, "sha256": _checkpoint_digest(payload)}
                )
        cases.append(case)
    return AnchorLibrary(
        version="3",
        lookback=lookback,
        horizon=horizon,
        seasonal_period=seasonal_period,
        target_col=dataset.target_col,
        timestamp_col=dataset.timestamp_col,
        dataset_name=dataset_name,
        model_source="expert_manifest" if expert_manifest else "builtin",
        cases=cases,
        clusters=build_anchor_clusters(cases, distance_weights=weights),
        expert_specs=expert_specs,
        distance_weights=weights,
        coverage=_coverage_metadata(
            source,
            origins,
            lookback,
            horizon,
            stride,
            start_row,
            end_row,
            geometry_case_ids=[case.case_id for case in cases],
            source_path=data_path,
        ),
    )


def _checkpoint_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _eligible_anchor_windows(
    source, lookback, horizon, stride, start_row, end_row, max_windows
):
    for name, value, minimum in (("lookback", lookback, 2), ("horizon", horizon, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    step = horizon if stride is None else stride
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("anchor stride must be a positive integer")
    cutoff = len(source.frame) if end_row is None else end_row
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int)
        or not 1 <= cutoff <= len(source.frame)
    ):
        raise ValueError(
            "end_row must be an exclusive row boundary within the source CSV"
        )
    first = lookback if start_row is None else start_row
    if isinstance(first, bool) or not isinstance(first, int) or first < 0:
        raise ValueError(
            "start_row must be an inclusive nonnegative target-origin boundary"
        )
    first = max(lookback, first)
    first = lookback + ((first - lookback + step - 1) // step) * step
    origins = list(range(first, cutoff - horizon + 1, step))
    if max_windows is not None:
        if (
            isinstance(max_windows, bool)
            or not isinstance(max_windows, int)
            or max_windows < 1
        ):
            raise ValueError("max_windows must be a positive integer")
        origins = _sample_origins(origins, max_windows)
    return (
        DatasetView(
            source.frame.iloc[:cutoff].copy(), source.target_col, source.timestamp_col
        ),
        origins,
    )


def _sample_origins(origins, count):
    if len(origins) <= count:
        return list(origins)
    if count == 1:
        return [origins[(len(origins) - 1) // 2]]
    indices = np.rint(np.linspace(0, len(origins) - 1, count)).astype(int)
    return [origins[int(index)] for index in indices]


def _window_at(dataset, origin, lookback, horizon):
    return TimeSeriesWindow(
        dataset.frame.iloc[origin - lookback : origin].copy(),
        dataset.frame.iloc[origin : origin + horizon].copy(),
        dataset.target_col,
        dataset.timestamp_col,
    )


def _validate_expert_fit_cutoffs(specs, first_origin, data_path):
    from backends.catalog import file_sha256

    source_hash = None
    for spec in specs:
        cutoff = spec.get("fit_end_row")
        if cutoff is None:
            continue
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < 1
            or first_origin < cutoff
        ):
            raise ValueError(
                f"expert {spec['name']!r} requires all case target origins after its fit_end_row"
            )
        expected = spec.get("source_sha256")
        if expected:
            source_hash = source_hash or file_sha256(data_path)
            if expected != source_hash:
                raise ValueError(
                    f"expert {spec['name']!r} was fitted on a different source CSV"
                )


def _coverage_metadata(
    source,
    origins,
    lookback,
    horizon,
    stride,
    start_row,
    end_row,
    *,
    geometry_case_ids,
    source_path=None,
):
    from backends.catalog import file_sha256

    timestamp = source.timestamp_col
    intervals = [
        {
            "target_start_row": origin,
            "target_end_row": origin + horizon,
            "history_start_row": origin - lookback,
            "target_start_timestamp": (
                str(source.frame[timestamp].iloc[origin]) if timestamp else None
            ),
            "target_end_timestamp": (
                str(source.frame[timestamp].iloc[origin + horizon - 1])
                if timestamp
                else None
            ),
        }
        for origin in origins
    ]
    cutoff = len(source.frame) if end_row is None else end_row
    return {
        "sampling": "uniform_eligible_origins_midpoint_if_one",
        "stride": horizon if stride is None else stride,
        "eligible_target_start_row": max(
            lookback, lookback if start_row is None else start_row
        ),
        "eligible_target_end_row_exclusive": cutoff,
        "source_rows": len(source.frame),
        "source_sha256": file_sha256(source_path) if source_path else None,
        "geometry_end_row_exclusive": cutoff,
        "geometry_case_ids": list(geometry_case_ids),
        "actual_target_intervals": intervals,
        "actual_target_start_row": min(origins) if origins else None,
        "actual_target_end_row_exclusive": max(origins) + horizon if origins else None,
    }


def expert_families(library: AnchorLibrary) -> dict[str, str]:
    families = {}
    for spec in library.expert_specs:
        family = spec.get("family", spec["name"])
        if not isinstance(family, str) or not family.strip():
            raise ValueError("expert family names must be nonempty strings")
        families[spec["name"]] = family
    for cluster in library.clusters:
        for name in cluster.model_errors:
            families.setdefault(name, name)
    return families


def assign_library_cluster(library: AnchorLibrary, history: Any) -> int:
    values = finite_vector(history, "anchor cluster query")
    if len(values) != library.lookback:
        raise ValueError("cluster query lookback differs from the fixed geometry")
    cluster = choose_cluster_by_similarity(
        library.clusters, values, distance_weights=library.distance_weights
    )
    if cluster is None:
        raise ValueError("fixed anchor geometry has no medoids")
    return cluster.cluster_id


def extend_library_cases(
    library: AnchorLibrary,
    data_path: str | Path,
    *,
    start_row: int,
    end_row: int,
    stride: int | None = None,
    max_windows: int | None = None,
    checkpoint_dir: str | Path | None = None,
    quiet_warnings: bool = False,
) -> AnchorLibrary:
    """Extend past reference coverage while preserving geometry and calibrated losses.

    ``max_windows`` is a total case cap, including the preserved geometry cases.
    Raw losses of new reference cases never replace geometry/calibration loss data.
    """
    result = copy.deepcopy(library)
    source = load_dataset(data_path, library.target_col, library.timestamp_col)
    dataset, eligible = _eligible_anchor_windows(
        source, library.lookback, library.horizon, stride, start_row, end_row, None
    )
    if any(case.target_start_row is None for case in result.cases):
        raise ValueError(
            "reference extension requires a bank with explicit case row boundaries"
        )
    from backends.catalog import file_sha256

    if library.coverage.get("source_sha256") not in {None, file_sha256(data_path)}:
        raise ValueError("reference extension source differs from the geometry source")
    existing = {case.target_start_row for case in result.cases}
    additions = [origin for origin in eligible if origin not in existing]
    if max_windows is not None:
        if (
            isinstance(max_windows, bool)
            or not isinstance(max_windows, int)
            or max_windows < len(result.cases)
        ):
            raise ValueError(
                "total reference case cap must accommodate all preserved geometry cases"
            )
        room = max_windows - len(result.cases)
        if additions and room == 0:
            raise ValueError(
                "total case cap leaves no room for later reference coverage"
            )
        additions = _sample_origins(additions, room) if room else []
    if additions:
        _validate_expert_fit_cutoffs(result.expert_specs, additions[0], data_path)
    checkpoints, identity = _prepare_case_checkpoints(
        checkpoint_dir,
        dataset,
        library.lookback,
        library.horizon,
        library.seasonal_period,
        library.dataset_name,
        stride,
        max_windows,
        result.expert_specs,
        library.distance_weights,
        additions,
    )
    next_id = max(case.case_id for case in result.cases) + 1
    for index, origin in enumerate(additions):
        window = _window_at(dataset, origin, library.lookback, library.horizon)
        path = (
            checkpoints / f"case_{index:08d}.json" if checkpoints is not None else None
        )
        if path is not None and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            payload = stored.get("payload") if isinstance(stored, dict) else None
            if (
                not isinstance(payload, dict)
                or stored.get("sha256") != _checkpoint_digest(payload)
                or payload.get("identity") != identity
            ):
                raise ValueError("reference case checkpoint is corrupt or incompatible")
            case = anchor_case_from_dict(payload["case"])
            if (
                case.target_start_row != origin
                or case.target_future != window.target_future.astype(float).tolist()
                or case.target_history != window.target_history.astype(float).tolist()
            ):
                raise ValueError(
                    "reference case checkpoint has incompatible source rows"
                )
        else:
            case = build_anchor_case(
                index,
                window,
                library.seasonal_period,
                dataset_name=library.dataset_name,
                quiet_warnings=quiet_warnings,
                expert_specs=result.expert_specs,
            )
            case.history_start_row, case.target_start_row, case.target_end_row = (
                origin - library.lookback,
                origin,
                origin + library.horizon,
            )
            if path is not None:
                payload = {"identity": identity, "case": asdict(case)}
                write_json(
                    path, {"payload": payload, "sha256": _checkpoint_digest(payload)}
                )
        case.case_id = next_id + index
        result.cases.append(case)
    geometry_ids = list(
        library.coverage.get(
            "geometry_case_ids", [case.case_id for case in library.cases]
        )
    )
    result.coverage = _coverage_metadata(
        source,
        sorted(case.target_start_row for case in result.cases),
        library.lookback,
        library.horizon,
        stride,
        min(library.coverage.get("eligible_target_start_row", start_row), start_row),
        end_row,
        geometry_case_ids=geometry_ids,
        source_path=data_path,
    )
    result.coverage["geometry_end_row_exclusive"] = library.coverage[
        "geometry_end_row_exclusive"
    ]
    result.coverage["sampling"] = "preserved_geometry_plus_uniform_later_references"
    result.coverage["reference_cases_added"] = len(additions)
    result.coverage["geometry_and_calibration_losses_unchanged"] = True
    for cluster in result.clusters:
        cluster.reference_case_ids = [
            case.case_id
            for case in result.cases
            if assign_library_cluster(result, case.target_history) == cluster.cluster_id
        ]
    result.version = "3"
    return result


def _prepare_case_checkpoints(
    directory,
    dataset,
    lookback,
    horizon,
    seasonal_period,
    dataset_name,
    stride,
    max_windows,
    expert_specs,
    distance_weights,
    origins,
):
    if directory is None:
        return None, None
    from backends import experts, catalog
    from anchorer import models
    import pandas as pd

    frame_hash = hashlib.sha256(
        pd.util.hash_pandas_object(dataset.frame, index=False).to_numpy().tobytes()
    ).hexdigest()
    identity = {
        "prefix_sha256": frame_hash,
        "prefix_rows": len(dataset.frame),
        "lookback": lookback,
        "horizon": horizon,
        "seasonal_period": seasonal_period,
        "dataset_name": dataset_name,
        "target_col": dataset.target_col,
        "timestamp_col": dataset.timestamp_col,
        "stride": stride,
        "max_windows": max_windows,
        "expert_specs": expert_specs,
        "distance_weights": list(distance_weights),
        "target_origins": origins,
        "implementation": {
            Path(path).name: catalog.file_sha256(path)
            for path in (__file__, experts.__file__, models.__file__)
        },
    }
    fingerprint = _checkpoint_digest(identity)
    output = Path(directory) / fingerprint
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text(encoding="utf-8")) != identity
    ):
        raise ValueError("anchor case checkpoint identity changed")
    write_json(identity_path, identity)
    return output, fingerprint


def write_json(path: Path, payload: Any) -> None:
    from utils.io import atomic_write_json

    atomic_write_json(path, payload)


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


def progress_iter(
    items: Sequence[T], *, label: str, unit: str, enabled: bool
) -> Iterator[T]:
    if not enabled:
        yield from items
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from items
        return
    yield from tqdm(items, total=len(items), desc=label, unit=unit)


def anchor_case_from_dict(payload: dict[str, Any]) -> AnchorCase:
    history = finite_vector(payload["target_history"], "anchor history")
    future = finite_vector(payload["target_future"], "anchor target")
    errors = dict(payload.get("model_errors", {}))
    if errors:
        validate_model_errors(errors)
    return AnchorCase(
        case_id=int(payload["case_id"]),
        target_history=history.tolist(),
        target_future=future.tolist(),
        window=finite_vector(
            payload.get("window", zscore(history)), "anchor representation"
        ).tolist(),
        best_model=str(payload["best_model"]),
        model_errors={str(k): float(v) for k, v in errors.items()},
        model_failures=dict(payload.get("model_failures", {})),
        history_start_row=payload.get("history_start_row"),
        target_start_row=payload.get("target_start_row"),
        target_end_row=payload.get("target_end_row"),
    )


def build_anchor_case(
    case_id: int,
    window: TimeSeriesWindow,
    seasonal_period: int,
    dataset_name: str | None = None,
    quiet_warnings: bool = False,
    expert_specs: list[dict[str, Any]] | None = None,
) -> AnchorCase:
    history = window.target_history
    future = window.target_future
    if future is None:
        raise ValueError("anchor cases require future target values")
    with anchorer_warning_context(quiet_warnings):
        if expert_specs is None:
            forecasts, failures = model_forecasts_for_window(
                window, len(future), seasonal_period, dataset_name
            )
        else:
            timestamps = (
                window.history[window.timestamp_col] if window.timestamp_col else None
            )
            forecasts, failures = forecast_expert_specs(
                expert_specs,
                history,
                len(future),
                seasonal_period,
                timestamps=timestamps,
            )
    if not forecasts:
        raise ValueError(
            f"no forecasting models succeeded for anchor case {case_id}; failures={failures}"
        )
    errors = {name: finite_mse(values, future) for name, values in forecasts.items()}
    best_model = min(errors, key=errors.get)
    return AnchorCase(
        case_id=case_id,
        target_history=np.asarray(history, dtype=float).tolist(),
        target_future=np.asarray(future, dtype=float).tolist(),
        window=zscore(history).tolist(),
        best_model=best_model,
        model_errors={name: float(value) for name, value in errors.items()},
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


def build_anchor_clusters(
    cases: list[AnchorCase],
    num_clusters: int = 6,
    *,
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> list[AnchorCluster]:
    if not cases:
        return []
    if (
        isinstance(num_clusters, bool)
        or not isinstance(num_clusters, int)
        or num_clusters < 1
    ):
        raise ValueError("number of anchor clusters must be a positive integer")
    k = min(num_clusters, len(cases))
    vectors = [case.window for case in cases]
    medoid_indices, cluster_indices = kmedoids_cluster(
        vectors, k, distance_weights=distance_weights
    )

    clusters: list[AnchorCluster] = []
    for cluster_id, medoid_idx in enumerate(medoid_indices):
        members = [cases[idx] for idx in cluster_indices[cluster_id]]
        counts: dict[str, int] = {}
        for member in members:
            counts[member.best_model] = counts.get(member.best_model, 0) + 1
        # Every expert in a cluster is evaluated on the same historical cases.
        names = set.intersection(*(set(member.model_errors) for member in members))
        if not names:
            raise ValueError(
                "cluster has no expert with a historical loss on every member; rebuild with a consistent expert pool"
            )
        losses = {
            name: float(np.mean([member.model_errors[name] for member in members]))
            for name in sorted(names)
        }
        validate_model_errors(losses)
        total_weight = sum(counts.values())
        clusters.append(
            AnchorCluster(
                cluster_id=cluster_id,
                window=cases[medoid_idx].window,
                best_model=counts,
                total_weight=total_weight,
                medoid_case_id=cases[medoid_idx].case_id,
                size=len(members),
                model_errors=losses,
                case_ids=[member.case_id for member in members],
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
    history = finite_vector(history, "anchor query")
    if library.promotion:
        from anchorer.policies import (
            validate_promotion_metadata,
            validate_promotion_query,
        )

        validate_promotion_metadata(library)
        validate_promotion_query(library, timestamps)
    if not library.cases:
        raise ValueError("anchor library has no cases")
    if (
        horizon != library.horizon
        or len(history) != library.lookback
        or seasonal_period != library.seasonal_period
    ):
        raise ValueError(
            "anchor query dimensions or seasonal period do not match its offline library"
        )
    cluster = choose_cluster_by_similarity(
        library.clusters, history, distance_weights=library.distance_weights
    )
    if cluster is None:
        raise ValueError("anchor library has no medoid clusters")
    validate_model_errors(cluster.model_errors)
    effective_losses = (
        cluster.weighting_losses if cluster.weighting_losses else cluster.model_errors
    )
    selected_names = sorted(effective_losses)
    if library.expert_specs:
        current_forecasts, failures = forecast_expert_specs(
            library.expert_specs,
            history,
            horizon,
            seasonal_period,
            timestamps=timestamps,
            model_names=selected_names,
            strict=True,
        )
    else:
        current_forecasts, failures = model_forecasts_from_history(
            history,
            horizon,
            seasonal_period,
            dataset_name=library.dataset_name,
            timestamps=timestamps,
            model_names=selected_names,
            strict=True,
        )
    if set(current_forecasts) != set(selected_names):
        missing = sorted(set(selected_names) - set(current_forecasts))
        raise ValueError(
            f"historically weighted experts are unavailable: {missing}; verify model dependencies and weights"
        )
    for name, values in current_forecasts.items():
        arr = finite_vector(values, f"expert {name!r}")
        if len(arr) != horizon:
            raise ValueError(f"expert {name!r} returned an incorrect forecast horizon")
    weights = historical_loss_weights(effective_losses)
    baseline = np.zeros(horizon, dtype=float)
    for name, weight in weights.items():
        if name in current_forecasts:
            baseline += weight * current_forecasts[name]

    return {
        "ensemble_forecast_baseline": baseline.tolist(),
        **(
            {
                "policy_selection": {
                    key: copy.deepcopy(library.promotion[key])
                    for key in (
                        "status",
                        "candidate_policy",
                        "selected_expert",
                        "selection_scope",
                        "promotion_start_row",
                        "promotion_labels_used_for_policy_selection",
                        "loss_fitting_used_promotion_labels",
                        "independent_quality_verified",
                    )
                }
            }
            if library.promotion
            else {}
        ),
        "reference_prediction": baseline.tolist(),
        "component_forecasts": {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in current_forecasts.items()
        },
        "weights": {name: float(value) for name, value in weights.items()},
        "weighting": "historical_loss_softmax",
        "historical_losses": dict(cluster.model_errors),
        "weighting_losses": dict(effective_losses),
        "loss_calibration": {
            key: copy.deepcopy(library.calibration[key])
            for key in (
                "method",
                "policy",
                "temperature",
                "shrinkage",
                "active_family_limit",
                "incumbent_name",
                "normalization_scale",
                "normalize_losses",
                "calibration_end_row",
                "prequential_mae_guard_passed",
            )
            if key in library.calibration
        },
        "best_model": {
            name: cluster.best_model.get(name, 0) for name in selected_names
        },
        "models": selected_names,
        "model_failures": failures,
        "selected_cluster": asdict(cluster) if cluster is not None else None,
        "anchor_cases": [
            {
                "case_id": case.case_id,
                "similarity": round(float(similarity), 6),
                "best_model": case.best_model,
                "model_errors": case.model_errors,
            }
            for case, similarity in ranked_anchor_cases(
                history,
                library.cases,
                top_k=top_k,
                distance_weights=library.distance_weights,
            )
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
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> list[tuple[AnchorCase, float]]:
    if top_k < 1:
        raise ValueError("anchor retrieval top_k must be positive")
    scored = [
        (
            case,
            comprehensive_similarity(
                history,
                np.asarray(case.target_history, dtype=float),
                distance_weights=distance_weights,
            ),
        )
        for case in cases
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, top_k)]


def model_weights_from_cases(cases: list[tuple[AnchorCase, float]]) -> dict[str, float]:
    if not cases:
        raise ValueError("historical cases are required to compute expert weights")
    names = set.intersection(*(set(case.model_errors) for case, _ in cases))
    losses = {
        name: float(np.mean([case.model_errors[name] for case, _ in cases]))
        for name in sorted(names)
    }
    return historical_loss_weights(losses)


def validate_model_errors(errors: dict[str, float]) -> None:
    if (
        not isinstance(errors, dict)
        or not errors
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 0
            for name, value in errors.items()
        )
    ):
        raise ValueError(
            "expert losses must be nonempty, finite, nonnegative historical losses"
        )


def historical_loss_weights(errors: dict[str, float]) -> dict[str, float]:
    validate_model_errors(errors)
    names = sorted(errors)
    losses = np.asarray([errors[name] for name in names], dtype=float)
    raw = np.exp(-(losses - float(np.min(losses))))
    weights = raw / float(np.sum(raw))
    return {name: float(value) for name, value in zip(names, weights)}


def default_model_weights(forecasts: dict[str, np.ndarray]) -> dict[str, float]:
    if not forecasts:
        raise ValueError("cannot weight an empty expert pool")
    value = 1.0 / len(forecasts)
    return {name: value for name in forecasts}


def model_forecasts_for_window(
    window: TimeSeriesWindow,
    horizon: int,
    seasonal_period: int,
    dataset_name: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    timestamps = (
        window.history[window.timestamp_col]
        if window.timestamp_col and window.timestamp_col in window.history
        else None
    )
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
    forecasts = primitive_model_forecasts(history, horizon, seasonal_period)
    if model_names is not None:
        if not model_names:
            raise ValueError("expert pool must not be empty")
        missing = set(model_names) - set(forecasts)
        if missing and strict:
            raise ValueError(
                f"local forecasting experts unavailable: {sorted(missing)}"
            )
        forecasts = {
            name: value for name, value in forecasts.items() if name in model_names
        }
    return forecasts, {}


def primitive_model_forecasts(
    history: np.ndarray, horizon: int, seasonal_period: int
) -> dict[str, np.ndarray]:
    y = finite_vector(history, "primitive expert history")
    if horizon < 1 or seasonal_period < 1:
        raise ValueError("horizon and seasonal_period must be positive")
    seasonal = seasonal_period
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


def kmedoids_cluster(
    vectors: list[list[float]],
    k: int,
    *,
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> tuple[list[int], list[list[int]]]:
    """Deterministic K-medoids using the same distance as online retrieval."""
    from functools import lru_cache

    if (
        not vectors
        or isinstance(k, bool)
        or not isinstance(k, int)
        or not 1 <= k <= len(vectors)
    ):
        raise ValueError("K-medoids requires 1 <= k <= number of windows")
    matrix = [zscore(finite_vector(vector, "cluster window")) for vector in vectors]
    weights = normalize_distance_weights(distance_weights)

    @lru_cache(maxsize=32768)
    def distance(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return combined_distance(matrix[i], matrix[j], weights, normalized=True)

    medoids = [0]
    while len(medoids) < k:
        remaining = [i for i in range(len(matrix)) if i not in medoids]
        medoids.append(
            max(
                remaining,
                key=lambda i: (
                    min(distance(min(i, j), max(i, j)) for j in medoids),
                    -i,
                ),
            )
        )
    for _ in range(100):
        groups: list[list[int]] = [[] for _ in medoids]
        for i in range(len(matrix)):
            group = (
                medoids.index(i)
                if i in medoids
                else min(
                    range(k),
                    key=lambda j: (distance(min(i, medoids[j]), max(i, medoids[j])), j),
                )
            )
            groups[group].append(i)
        updated = [
            min(
                group,
                key=lambda i: (sum(distance(min(i, j), max(i, j)) for j in group), i),
            )
            for group in groups
        ]
        if updated == medoids:
            return medoids, groups
        medoids = updated
    raise RuntimeError("K-medoids did not converge within 100 iterations")


def evenly_spaced_indices(size: int, k: int) -> list[int]:
    if k <= 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, size - 1, k)]


def choose_cluster_by_similarity(
    clusters: list[AnchorCluster],
    history: np.ndarray,
    *,
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> AnchorCluster | None:
    if not clusters:
        return None
    return min(
        clusters,
        key=lambda cluster: combined_distance(
            history, np.asarray(cluster.window, dtype=float), distance_weights
        ),
    )


def finite_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    value = mse(prediction, target)
    if not np.isfinite(value):
        raise ValueError("historical expert loss must be finite")
    return float(value)


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


def auto_arima_forecast(
    y: np.ndarray, horizon: int, seasonal_period: int
) -> np.ndarray:
    from backends.experts import auto_arima_forecast as predict

    return predict(y, horizon, seasonal_period)


def theta_forecast(y: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel

    period = (
        int(seasonal_period)
        if seasonal_period > 1 and len(y) >= seasonal_period * 2
        else None
    )
    fitted = ThetaModel(y, period=period).fit()
    return np.asarray(fitted.forecast(horizon), dtype=float)


def comprehensive_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    return float(1.0 / (1.0 + combined_distance(left, right, distance_weights)))


def normalize_distance_weights(values: Any) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if (
        arr.shape != (3,)
        or not np.isfinite(arr).all()
        or np.any(arr < 0)
        or np.max(arr) <= 0
    ):
        raise ValueError(
            "anchor distance weights must be three finite nonnegative values with positive sum"
        )
    arr = arr / np.max(arr)
    arr = arr / np.sum(arr)
    return tuple(float(v) for v in arr)


def combined_distance(
    left: np.ndarray,
    right: np.ndarray,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    *,
    normalized: bool = False,
) -> float:
    a, b = finite_vector(left, "distance input"), finite_vector(right, "distance input")
    if len(a) != len(b):
        raise ValueError("anchor representations must have the same lookback length")
    if not normalized:
        a, b = zscore(a), zscore(b)
    wd, we, wc = normalize_distance_weights(weights)
    if np.array_equal(a, b):
        return 0.0
    scale = np.sqrt(len(a))
    euclidean = float(np.linalg.norm(a - b) / scale)
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine_distance = (
        (0.0 if np.array_equal(a, b) else 1.0)
        if norms <= 1e-15
        else 1.0 - float(np.clip(np.dot(a, b) / norms, -1, 1))
    )
    dtw_value = dtw_distance(a, b) / scale if wd else 0.0
    return float(wd * dtw_value + we * euclidean + wc * cosine_distance)


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    a, b = (
        np.ascontiguousarray(finite_vector(left, "DTW input"), dtype=np.double),
        np.ascontiguousarray(finite_vector(right, "DTW input"), dtype=np.double),
    )
    if np.array_equal(a, b):
        return 0.0
    try:
        from dtaidistance import dtw

        return float(dtw.distance_fast(a, b, use_pruning=True))
    except (ImportError, AttributeError, RuntimeError):
        pass
    # Anti-diagonal cells are independent; this also defines the exact fallback.
    n, m = len(a), len(b)
    costs = np.full((n + 1, m + 1), np.inf, dtype=float)
    costs[0, 0] = 0.0
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        j = diagonal - i
        costs[i, j] = (a[i - 1] - b[j - 1]) ** 2 + np.minimum(
            np.minimum(costs[i - 1, j], costs[i, j - 1]), costs[i - 1, j - 1]
        )
    return float(np.sqrt(costs[n, m]))


def align_length(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(values) == len(reference):
        return values
    if len(values) > len(reference):
        return values[-len(reference) :]
    pad = np.repeat(values[0], len(reference) - len(values))
    return np.concatenate([pad, values])


def zscore(values: np.ndarray) -> np.ndarray:
    values = finite_vector(values, "z-score input")
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return np.zeros_like(values, dtype=float)
    values = values / scale
    std = float(np.std(values))
    if std == 0:
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
