"""Past-only prequential calibration of historical expert losses and family choice."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from anchorer.build import (
    AnchorLibrary,
    expert_families,
    historical_loss_weights,
    validate_model_errors,
)
from backends.catalog import content_fingerprint
from backends.experts import finite_vector
from utils.metrics import mae, mse


class CalibrationError(ValueError):
    """An infeasible calibration with inspectable, serializable diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def fit_library_calibration(
    library: AnchorLibrary,
    historical_prediction_records: Sequence[dict[str, Any]],
    *,
    source_sha256: str,
    calibration_start_row: int,
    calibration_end_row: int,
    confirmation_start_row: int,
    temperatures: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0),
    shrinkages: Sequence[float] = (0.0, 4.0, 16.0),
    initial_loss_records: Sequence[dict[str, Any]] | None = None,
    normalize_losses: bool = True,
    scale_floor: float = 1e-12,
    active_family_limits: Sequence[int | None] = (1, 2, 3, None),
    incumbent_name: str | None = None,
    allow_unqualified: bool = False,
) -> AnchorLibrary:
    """Return a new calibrated bank; medoids and every raw geometry loss stay intact.

    Each prequential score is formed from geometry seeds and completed earlier
    calibration targets. Its current label cannot choose a family variant or its
    weight. The final transform is refitted on this past prefix only, before any
    confirmation target. ``weighting_losses`` remain the arguments of softmax(-L).
    """
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", source_sha256
    ):
        raise ValueError("calibration requires an explicit source SHA256")
    for name, value in (
        ("calibration_start_row", calibration_start_row),
        ("calibration_end_row", calibration_end_row),
        ("confirmation_start_row", confirmation_start_row),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not calibration_start_row < calibration_end_row <= confirmation_start_row:
        raise ValueError("calibration targets must precede confirmation")
    geometry_end = library.coverage.get("geometry_end_row_exclusive")
    if (
        isinstance(geometry_end, bool)
        or not isinstance(geometry_end, int)
        or geometry_end > calibration_start_row
    ):
        raise ValueError(
            "calibration requires fixed geometry ending before calibration starts"
        )
    if library.coverage.get("source_sha256") != source_sha256:
        raise ValueError("calibration source differs from the fixed geometry source")
    if (
        not isinstance(normalize_losses, bool)
        or isinstance(scale_floor, bool)
        or not isinstance(scale_floor, (int, float))
        or not math.isfinite(scale_floor)
        or scale_floor <= 0
    ):
        raise ValueError(
            "normalization requires a boolean switch and a finite positive numerical floor"
        )
    temperature_values = _parameters(temperatures, "temperature", positive=True)
    shrinkage_values = _parameters(shrinkages, "shrinkage", positive=False)
    model_names = sorted(library.clusters[0].model_errors) if library.clusters else []
    if not model_names or any(
        set(cluster.model_errors) != set(model_names) for cluster in library.clusters
    ):
        raise ValueError(
            "calibration requires the same complete expert pool in every geometry cluster"
        )
    families = expert_families(library)
    family_limits = _family_limits(
        active_family_limits, len(set(families[name] for name in model_names))
    )
    if not isinstance(allow_unqualified, bool):
        raise ValueError("allow_unqualified must be an explicit boolean")
    if incumbent_name is not None and incumbent_name not in model_names:
        raise ValueError("the frozen incumbent must name a retained expert")
    cluster_ids = {cluster.cluster_id for cluster in library.clusters}
    if len(cluster_ids) != len(library.clusters):
        raise ValueError("geometry cluster IDs must be unique")
    seeds = (
        _geometry_seeds(library, model_names, geometry_end)
        if initial_loss_records is None
        else _validate_seed_records(
            initial_loss_records,
            model_names,
            cluster_ids,
            geometry_end,
            library.horizon,
        )
    )
    if not seeds:
        raise ValueError("prequential calibration needs historical geometry loss seeds")
    records = _validate_prediction_records(
        historical_prediction_records,
        model_names,
        cluster_ids,
        library.horizon,
        calibration_start_row,
        calibration_end_row,
    )
    supplied_incumbent = ["incumbent_prediction" in record for record in records]
    if any(supplied_incumbent) and not all(supplied_incumbent):
        raise ValueError("fixed-incumbent traces must cover every calibration origin")
    if incumbent_name is not None:
        for record in records:
            declared = record.get(
                "incumbent_prediction", record["predictions"][incumbent_name]
            )
            if not np.allclose(
                declared, record["predictions"][incumbent_name], rtol=1e-10, atol=1e-10
            ):
                raise ValueError(
                    "incumbent trace differs from its retained expert prediction"
                )
            record["incumbent_prediction"] = record["predictions"][incumbent_name]
    incumbent_metrics = None
    if all("incumbent_prediction" in record for record in records):
        incumbent_metrics = {
            "mse": _mean(
                [
                    mse(record["incumbent_prediction"], record["target"])
                    for record in records
                ]
            ),
            "mae": _mean(
                [
                    mae(record["incumbent_prediction"], record["target"])
                    for record in records
                ]
            ),
        }
    trials = [
        _prequential_trial(
            records,
            seeds,
            model_names,
            families,
            library.horizon,
            temperature,
            shrinkage,
            limit,
            normalize_losses,
            scale_floor,
        )
        for temperature in temperature_values
        for shrinkage in shrinkage_values
        for limit in family_limits
    ]
    if incumbent_name is not None:
        trials.append(
            _prequential_trial(
                records,
                seeds,
                model_names,
                families,
                library.horizon,
                1.0,
                0.0,
                1,
                normalize_losses,
                scale_floor,
                incumbent_name=incumbent_name,
            )
        )
    eligible = list(range(len(trials)))
    if incumbent_metrics is not None:
        tolerance = 1e-10 * max(1.0, abs(incumbent_metrics["mae"]))
        eligible = [
            index
            for index, trial in enumerate(trials)
            if trial["mae"] <= incumbent_metrics["mae"] + tolerance
        ]
    guard_passed = bool(eligible) if incumbent_metrics is not None else None
    if not eligible:
        if not allow_unqualified:
            raise CalibrationError(
                "no prequential candidate meets the frozen-incumbent MAE guard",
                {"incumbent": incumbent_metrics, "prequential_trials": trials},
            )
        eligible = list(range(len(trials)))
    chosen_index = min(
        eligible,
        key=lambda index: (
            trials[index]["mse"],
            trials[index]["mae"],
            0 if trials[index]["policy"] == "fixed_incumbent" else 1,
            index,
        ),
    )
    chosen = trials[chosen_index]
    final_rows = seeds + [_loss_record(record) for record in records]
    result = copy.deepcopy(library)
    cluster_statistics = {}
    for cluster in result.clusters:
        losses, statistics = _policy_losses(
            final_rows,
            cluster.cluster_id,
            model_names,
            families,
            chosen["shrinkage"],
            chosen["temperature"],
            normalize_losses,
            scale_floor,
            chosen["active_family_limit"],
            incumbent_name if chosen["policy"] == "fixed_incumbent" else None,
        )
        cluster.weighting_losses = losses
        cluster.calibration_count = statistics["local_records"]
        cluster_statistics[str(cluster.cluster_id)] = statistics
    result.version = "3"
    result.calibration = {
        "method": "prequential_historical_validation_loss_softmax",
        "loss_transform": "L = shrunken_past_validation_MSE / (past_seasonal_naive_MSE * temperature)"
        if normalize_losses
        else "L = shrunken_past_validation_MSE / temperature",
        "temperature": chosen["temperature"],
        "shrinkage": chosen["shrinkage"],
        "policy": chosen["policy"],
        "active_family_limit": chosen["active_family_limit"],
        "incumbent_name": incumbent_name,
        "prequential_incumbent_metrics": incumbent_metrics,
        "prequential_mae_guard_passed": guard_passed,
        "prequential_mae_guard": "unavailable_no_fixed_incumbent_trace"
        if incumbent_metrics is None
        else "fixed_before_calibration_incumbent",
        "allow_unqualified": allow_unqualified,
        "normalize_losses": normalize_losses,
        "scale_floor": scale_floor,
        "normalization_scale": next(iter(cluster_statistics.values()))[
            "normalization_scale"
        ],
        "family_rule": "fixed_before_calibration_incumbent_singleton"
        if chosen["policy"] == "fixed_incumbent"
        else "one minimum estimated-past-loss variant per family and cluster",
        "temperature_candidates": temperature_values,
        "shrinkage_candidates": shrinkage_values,
        "active_family_limit_candidates": family_limits,
        "families": families,
        "source_sha256": source_sha256,
        "geometry_end_row_exclusive": geometry_end,
        "initial_geometry_case_ids": list(
            library.coverage.get(
                "geometry_case_ids", [case.case_id for case in library.cases]
            )
        ),
        "initial_geometry_records": len(seeds),
        "calibration_records": len(records),
        "calibration_start_row": calibration_start_row,
        "calibration_end_row": calibration_end_row,
        "confirmation_start_row": confirmation_start_row,
        "actual_calibration_origins": [record["origin"] for record in records],
        "actual_calibration_target_end_row": max(
            record["target_end_row"] for record in records
        ),
        "prediction_records_sha256": content_fingerprint(
            _serializable_records(records)
        ),
        "initial_loss_records_sha256": content_fingerprint(seeds),
        "clusters": cluster_statistics,
        "prequential_trials": trials,
        "selected_trial_index": chosen_index,
        "geometry_unchanged": True,
        "confirmation_labels_used": False,
    }
    return result


def _family_limits(values, family_count):
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ValueError("active family limits must be a nonempty sequence")
    result = []
    for value in values:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError("active family limits must be positive integers or None")
        effective = None if value is None or value >= family_count else value
        if effective not in result:
            result.append(effective)
    return result


def _prequential_trial(
    records,
    seeds,
    names,
    families,
    horizon,
    temperature,
    shrinkage,
    limit,
    normalize,
    floor,
    incumbent_name=None,
):
    past, scores, trace = list(seeds), [], []
    for record in records:
        if any(row["target_end_row"] > record["origin"] for row in past):
            raise ValueError(
                "overlapping targets would leak into a prequential loss estimate"
            )
        losses, statistics = _policy_losses(
            past,
            record["cluster_id"],
            names,
            families,
            shrinkage,
            temperature,
            normalize,
            floor,
            limit,
            incumbent_name,
        )
        weights = historical_loss_weights(losses)
        prediction = np.zeros(horizon)
        for name, weight in weights.items():
            prediction += weight * record["predictions"][name]
        scores.append(
            {
                "mse": mse(prediction, record["target"]),
                "mae": mae(prediction, record["target"]),
            }
        )
        trace.append(
            {
                "origin": record["origin"],
                "cluster_id": record["cluster_id"],
                "past_records": len(past),
                "latest_past_target_end_row": max(
                    row["target_end_row"] for row in past
                ),
                "normalization_scale": statistics["normalization_scale"],
                "active_variants": statistics["active_variants"],
                "weights": weights,
            }
        )
        past.append(_loss_record(record))
    return {
        "policy": "fixed_incumbent" if incumbent_name else "historical_loss",
        "temperature": temperature,
        "shrinkage": shrinkage,
        "active_family_limit": limit,
        "mse": _mean([score["mse"] for score in scores]),
        "mae": _mean([score["mae"] for score in scores]),
        "scored_origins": [record["origin"] for record in records],
        "trace": trace,
    }


def _policy_losses(
    records,
    cluster_id,
    names,
    families,
    shrinkage,
    temperature,
    normalize,
    floor,
    limit,
    incumbent_name,
):
    losses, statistics = _cluster_losses(
        records,
        cluster_id,
        names,
        families,
        shrinkage,
        temperature,
        normalize,
        floor,
        limit,
    )
    if incumbent_name is not None:
        losses = {
            incumbent_name: statistics["shrunken_validation_mse"][incumbent_name]
            / statistics["normalization_scale"]
            / temperature
        }
        validate_model_errors(losses)
        statistics["active_variants"] = {families[incumbent_name]: incumbent_name}
        statistics["weighting_losses"] = losses
    return losses, statistics


def _parameters(values, name, *, positive):
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise ValueError(f"{name} candidates must be a nonempty sequence")
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(
                f"{name} candidates must be finite and {'positive' if positive else 'nonnegative'}"
            )
        if float(value) in result:
            raise ValueError(f"duplicate {name} candidate")
        result.append(float(value))
    return result


def _geometry_seeds(library, model_names, geometry_end):
    geometry_ids = library.coverage.get(
        "geometry_case_ids", [case.case_id for case in library.cases]
    )
    if (
        not isinstance(geometry_ids, list)
        or not geometry_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in geometry_ids
        )
        or len(geometry_ids) != len(set(geometry_ids))
    ):
        raise ValueError("geometry case IDs must be explicit, complete, and unique")
    lookup = {case.case_id: case for case in library.cases}
    assignment = {}
    for cluster in library.clusters:
        for case_id in cluster.case_ids:
            if case_id in assignment:
                raise ValueError("geometry case belongs to multiple medoid clusters")
            assignment[case_id] = cluster.cluster_id
    seeds = []
    for case_id in geometry_ids:
        if case_id not in lookup or case_id not in assignment:
            raise ValueError(
                "geometry seed case or its fixed cluster assignment is missing"
            )
        case = lookup[case_id]
        if case.target_start_row is None or case.target_end_row is None:
            raise ValueError(
                "geometry seed cases require explicit temporal row boundaries"
            )
        history = finite_vector(case.target_history, "geometry seed history")
        prediction = np.resize(
            history[-min(len(history), library.seasonal_period) :], library.horizon
        )
        seeds.append(
            {
                "origin": case.target_start_row,
                "target_end_row": case.target_end_row,
                "cluster_id": assignment[case_id],
                "model_errors": dict(case.model_errors),
                "naive_mse": mse(prediction, case.target_future),
            }
        )
    return _validate_seed_records(
        seeds,
        model_names,
        {cluster.cluster_id for cluster in library.clusters},
        geometry_end,
        library.horizon,
    )


def _validate_seed_records(records, model_names, cluster_ids, geometry_end, horizon):
    if (
        isinstance(records, (str, bytes, dict))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ValueError("initial loss records must be a nonempty sequence")
    result = []
    seen = set()
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("initial loss record must be an object")
        row = copy.deepcopy(source)
        origin = row.get("origin")
        end = row.get(
            "target_end_row",
            origin + horizon
            if isinstance(origin, int) and not isinstance(origin, bool)
            else None,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (origin, end)
            )
            or origin < 0
            or end <= origin
            or end > geometry_end
        ):
            raise ValueError("initial losses may only use completed geometry targets")
        if (
            end != origin + horizon
            or origin in seen
            or isinstance(row.get("cluster_id"), bool)
            or not isinstance(row.get("cluster_id"), int)
            or row.get("cluster_id") not in cluster_ids
        ):
            raise ValueError("duplicate seed origin or unknown geometry cluster")
        seen.add(origin)
        validate_model_errors(row.get("model_errors"))
        if set(row["model_errors"]) != set(model_names):
            raise ValueError("initial geometry losses must cover every retained expert")
        if (
            isinstance(row.get("naive_mse"), bool)
            or not isinstance(row.get("naive_mse"), (int, float))
            or not math.isfinite(row["naive_mse"])
            or row["naive_mse"] < 0
        ):
            raise ValueError(
                "initial seasonal-naive loss must be finite and nonnegative"
            )
        result.append(
            {
                "origin": origin,
                "target_end_row": end,
                "cluster_id": row["cluster_id"],
                "model_errors": {
                    name: float(row["model_errors"][name]) for name in model_names
                },
                "naive_mse": float(row["naive_mse"]),
            }
        )
    return sorted(result, key=lambda row: row["origin"])


def _validate_prediction_records(
    records, model_names, cluster_ids, horizon, start, end
):
    if (
        isinstance(records, (str, bytes, dict))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ValueError("calibration predictions must be a nonempty sequence")
    result = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("calibration prediction record must be an object")
        origin = source.get("origin")
        if (
            isinstance(origin, bool)
            or not isinstance(origin, int)
            or origin < start
            or origin + horizon > end
        ):
            raise ValueError(
                "calibration target rows lie outside the declared past interval"
            )
        if (
            source.get("target_end_row", origin + horizon) != origin + horizon
            or isinstance(source.get("cluster_id"), bool)
            or not isinstance(source.get("cluster_id"), int)
            or source.get("cluster_id") not in cluster_ids
        ):
            raise ValueError("calibration record has an invalid horizon or cluster")
        predictions = source.get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != set(model_names):
            raise ValueError(
                "calibration trace must include exactly every retained expert"
            )
        row = {
            "origin": origin,
            "target_end_row": origin + horizon,
            "cluster_id": source["cluster_id"],
            "predictions": {},
        }
        for key in ("target", "naive_prediction"):
            values = finite_vector(source.get(key), f"calibration {key}")
            if len(values) != horizon:
                raise ValueError("calibration arrays must cover the complete horizon")
            row[key] = values
        for name in model_names:
            values = finite_vector(predictions[name], f"calibration expert {name}")
            if len(values) != horizon:
                raise ValueError(
                    "expert calibration prediction has an incomplete horizon"
                )
            row["predictions"][name] = values
        if "incumbent_prediction" in source:
            values = finite_vector(
                source["incumbent_prediction"], "fixed incumbent calibration trace"
            )
            if len(values) != horizon:
                raise ValueError("fixed incumbent trace has an incomplete horizon")
            row["incumbent_prediction"] = values
        _loss_record(row)
        result.append(row)
    result.sort(key=lambda row: row["origin"])
    if any(
        left["target_end_row"] > right["origin"]
        for left, right in zip(result, result[1:])
    ):
        raise ValueError(
            "calibration origins must be unique with nonoverlapping target intervals"
        )
    return result


def _loss_record(row):
    return {
        "origin": row["origin"],
        "target_end_row": row["target_end_row"],
        "cluster_id": row["cluster_id"],
        "model_errors": {
            name: mse(prediction, row["target"])
            for name, prediction in row["predictions"].items()
        },
        "naive_mse": mse(row["naive_prediction"], row["target"]),
    }


def _cluster_losses(
    records,
    cluster_id,
    model_names,
    families,
    shrinkage,
    temperature,
    normalize,
    floor,
    family_limit=None,
):
    if not records:
        raise ValueError("past validation losses are required to form weights")
    local = [row for row in records if row["cluster_id"] == cluster_id]
    global_mse = {
        name: _mean([row["model_errors"][name] for row in records])
        for name in model_names
    }
    local_mse = (
        {
            name: _mean([row["model_errors"][name] for row in local])
            for name in model_names
        }
        if local
        else {}
    )
    local_weight = len(local) / (len(local) + shrinkage) if local else 0.0
    effective = {
        name: local_weight * local_mse[name] + (1 - local_weight) * global_mse[name]
        if local
        else global_mse[name]
        for name in model_names
    }
    active = {}
    for name in sorted(model_names):
        family = families[name]
        if family not in active or (effective[name], name) < (
            effective[active[family]],
            active[family],
        ):
            active[family] = name
    if family_limit is not None:
        selected = sorted(
            active, key=lambda family: (effective[active[family]], active[family])
        )[:family_limit]
        active = {family: active[family] for family in selected}
    naive_mse = _mean([row["naive_mse"] for row in records])
    scale = max(naive_mse, floor) if normalize else 1.0
    losses = {
        name: effective[name] / scale / temperature for name in sorted(active.values())
    }
    validate_model_errors(losses)
    return losses, {
        "local_records": len(local),
        "global_records": len(records),
        "raw_local_mse": local_mse,
        "raw_global_mse": global_mse,
        "shrunken_validation_mse": effective,
        "historical_seasonal_naive_mse": naive_mse,
        "normalization_scale": scale,
        "active_variants": active,
        "weighting_losses": losses,
    }


def _mean(values):
    return math.fsum(float(value) / len(values) for value in values)


def _serializable_records(records):
    return [
        {
            **record,
            "target": record["target"].tolist(),
            "naive_prediction": record["naive_prediction"].tolist(),
            **(
                {"incumbent_prediction": record["incumbent_prediction"].tolist()}
                if "incumbent_prediction" in record
                else {}
            ),
            "predictions": {
                name: value.tolist() for name, value in record["predictions"].items()
            },
        }
        for record in records
    ]
