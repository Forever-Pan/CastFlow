"""Construct frozen policy alternatives without consulting promotion labels."""

from __future__ import annotations

import copy
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from anchorer.build import AnchorLibrary, validate_model_errors
from anchorer.calibration import _serializable_records, _validate_prediction_records
from backends.catalog import content_fingerprint
from utils.metrics import mae, mse


POLICIES = ("mixture", "calibration_default_champion", "selection_incumbent")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value: Any, name: str) -> None:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} must be a full SHA256",
    )


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}",
    )
    return float(value)


def _default_names(library: AnchorLibrary) -> list[str]:
    # Initializer retention marks original defaults explicitly. An older library
    # without this provenance cannot safely infer which tuned recipe is default.
    names = [
        spec["name"]
        for spec in library.expert_specs
        if spec.get("variant_role") == "default"
    ]
    _require(
        bool(names),
        "promotion needs explicitly retained original-default experts; rebuild the library",
    )
    families = [
        next(
            spec.get("family", spec["name"])
            for spec in library.expert_specs
            if spec["name"] == name
        )
        for name in names
    ]
    _require(
        len(families) == len(set(families)),
        "original-default expert families are ambiguous",
    )
    return names


def _singleton_loss(library: AnchorLibrary, cluster_id: int, name: str) -> float:
    details = library.calibration["clusters"][str(cluster_id)]
    error = _number(
        details["shrunken_validation_mse"][name], "historical validation MSE"
    )
    scale = _number(
        details["normalization_scale"], "historical normalization scale", positive=True
    )
    temperature = _number(
        library.calibration["temperature"], "historical temperature", positive=True
    )
    value = error / scale / temperature
    _number(value, "singleton historical weighting loss")
    return value


def build_promotion_candidates(
    library: AnchorLibrary,
    *,
    calibration_records: list[dict[str, Any]],
    selection_incumbent_name: str,
    pipeline_fingerprint: str,
) -> dict[str, AnchorLibrary]:
    """Freeze the mixture and two historical singleton alternatives.

    The default champion is selected using complete fixed-expert predictions on
    the earlier calibration interval. No prediction or target from the later
    promotion interval is accepted. Numerical historical losses and geometry
    remain intact; a singleton restricts only its associated expert set.
    """
    _sha(pipeline_fingerprint, "pipeline fingerprint")
    _require(
        library.version == "3" and bool(library.calibration),
        "promotion requires a calibrated version-3 library",
    )
    _require(
        not library.promotion,
        "promotion candidates must start from the original pre-promotion library",
    )
    metadata = library.calibration
    _require(
        metadata.get("confirmation_labels_used") is False,
        "the base calibration must precede promotion labels",
    )
    _require(
        selection_incumbent_name == metadata.get("incumbent_name"),
        "selection incumbent differs from the frozen pre-calibration recipe",
    )
    names = sorted(spec["name"] for spec in library.expert_specs)
    defaults = _default_names(library)
    _require(
        selection_incumbent_name in defaults,
        "selection incumbent must be a retained original default",
    )
    rows = _validate_prediction_records(
        calibration_records,
        names,
        {cluster.cluster_id for cluster in library.clusters},
        library.horizon,
        metadata["calibration_start_row"],
        metadata["calibration_end_row"],
    )
    for row in rows:
        if "incumbent_prediction" in row:
            _require(
                row["incumbent_prediction"].tolist()
                == row["predictions"][selection_incumbent_name].tolist(),
                "calibration incumbent trace differs from its frozen expert",
            )
        row["incumbent_prediction"] = row["predictions"][selection_incumbent_name]
    records_hash = content_fingerprint(_serializable_records(rows))
    _require(
        records_hash == metadata["prediction_records_sha256"],
        "promotion policy construction requires the exact frozen calibration traces",
    )
    _require(
        [row["origin"] for row in rows] == metadata["actual_calibration_origins"],
        "calibration origin coverage differs from the frozen library",
    )
    scores = [
        {
            "name": name,
            "mse": math.fsum(
                mse(row["predictions"][name], row["target"]) for row in rows
            )
            / len(rows),
            "mae": math.fsum(
                mae(row["predictions"][name], row["target"]) for row in rows
            )
            / len(rows),
        }
        for name in defaults
    ]
    winner = min(
        range(len(scores)),
        key=lambda index: (scores[index]["mse"], scores[index]["mae"], index),
    )
    champion = scores[winner]
    common = {
        "schema_version": 1,
        "status": "candidate",
        "pipeline_fingerprint": pipeline_fingerprint,
        "parent_library_fingerprint": content_fingerprint(library.to_dict()),
        "calibration_records_sha256": records_hash,
        "calibration_default_champion": copy.deepcopy(champion),
        "calibration_default_scores": scores,
        "calibration_default_selection": "fixed predictions on complete earlier calibration interval; MSE then MAE then retained order",
        "selection_incumbent_name": selection_incumbent_name,
        "calibration_start_row": metadata["calibration_start_row"],
        "calibration_end_row": metadata["calibration_end_row"],
        "promotion_start_row": metadata["confirmation_start_row"],
        "selection_scope": "training_side_promotion_validation",
        "promotion_labels_used_for_policy_selection": False,
        "loss_fitting_used_promotion_labels": False,
        "independent_quality_verified": False,
        "base_calibration_scope": "unchanged historical loss fitting before promotion; its confirmation_labels_used flag describes that fitting only",
        "original_calibrated_active_losses": {
            str(cluster.cluster_id): dict(cluster.weighting_losses)
            for cluster in library.clusters
        },
    }
    result = {}
    for policy, expert in zip(
        POLICIES, (None, champion["name"], selection_incumbent_name)
    ):
        candidate = copy.deepcopy(library)
        candidate.promotion = {
            **copy.deepcopy(common),
            "candidate_policy": policy,
            "selected_expert": expert,
        }
        if expert is not None:
            for cluster in candidate.clusters:
                cluster.weighting_losses = {
                    expert: _singleton_loss(library, cluster.cluster_id, expert)
                }
        validate_promotion_metadata(candidate)
        result[policy] = candidate
    return result


def validate_promotion_metadata(library: AnchorLibrary) -> None:
    """Reject ambiguous label use or policy/active-expert inconsistencies."""
    value = library.promotion
    _require(
        library.version == "3" and bool(library.calibration),
        "policy promotion requires a calibrated version-3 library",
    )
    _require(
        type(value.get("schema_version")) is int and value["schema_version"] == 1,
        "unsupported promotion metadata version",
    )
    _require(
        value.get("status") in {"candidate", "selected"}, "invalid promotion status"
    )
    _require(
        value.get("candidate_policy") in POLICIES, "unknown promotion candidate policy"
    )
    _require(
        value.get("selection_scope") == "training_side_promotion_validation",
        "promotion must declare its training-side model-selection scope",
    )
    _require(
        value.get("independent_quality_verified") is False
        and value.get("loss_fitting_used_promotion_labels") is False,
        "promotion selection cannot claim independent quality or later-label loss fitting",
    )
    _require(
        value.get("promotion_labels_used_for_policy_selection")
        is (value["status"] == "selected"),
        "promotion label-use declaration disagrees with its status",
    )
    if value["status"] == "selected":
        _require(
            value.get("selection_policy") == "conservative",
            "selected promotion policy must be explicit",
        )
        _require(
            bool(library.timestamp_col),
            "selected promotion policies require dated histories",
        )
        end = value.get("promotion_end_row_exclusive")
        _require(
            type(end) is int
            and type(value.get("promotion_start_row")) is int
            and end > value["promotion_start_row"],
            "selected promotion interval must have an exclusive end",
        )
        _promotion_timestamp(value.get("promotion_last_timestamp"))
    for name in (
        "pipeline_fingerprint",
        "parent_library_fingerprint",
        "calibration_records_sha256",
    ):
        _sha(value.get(name), name)
    for key in ("calibration_start_row", "calibration_end_row", "promotion_start_row"):
        actual = value.get(key)
        expected = library.calibration[
            "confirmation_start_row" if key == "promotion_start_row" else key
        ]
        _require(
            type(actual) is int and actual == expected,
            "promotion and historical calibration boundaries differ",
        )
    _require(
        value["calibration_start_row"]
        < value["calibration_end_row"]
        <= value["promotion_start_row"],
        "promotion label interval overlaps loss fitting",
    )
    _require(
        value["calibration_records_sha256"]
        == library.calibration["prediction_records_sha256"],
        "promotion calibration identity differs",
    )
    defaults = _default_names(library)
    scores = value.get("calibration_default_scores")
    _require(
        isinstance(scores, list)
        and [row.get("name") for row in scores if isinstance(row, dict)] == defaults,
        "calibration default score coverage differs",
    )
    for row in scores:
        _number(row.get("mse"), "default calibration MSE")
        _number(row.get("mae"), "default calibration MAE")
    champion = scores[
        min(
            range(len(scores)),
            key=lambda index: (scores[index]["mse"], scores[index]["mae"], index),
        )
    ]
    _require(
        value.get("calibration_default_champion") == champion,
        "declared calibration champion does not follow its frozen selection rule",
    )
    _require(
        value.get("selection_incumbent_name")
        == library.calibration.get("incumbent_name")
        and value["selection_incumbent_name"] in defaults,
        "original selection incumbent changed",
    )
    original = value.get("original_calibrated_active_losses")
    _require(
        isinstance(original, dict)
        and set(original) == {str(cluster.cluster_id) for cluster in library.clusters},
        "original calibrated policy coverage is incomplete",
    )
    expert = value.get("selected_expert")
    expected = {
        "mixture": None,
        "calibration_default_champion": champion["name"],
        "selection_incumbent": value["selection_incumbent_name"],
    }[value["candidate_policy"]]
    _require(expert == expected, "selected expert does not match the frozen candidate")
    for cluster in library.clusters:
        losses = original[str(cluster.cluster_id)]
        validate_model_errors(losses)
        _require(
            set(losses).issubset(cluster.model_errors),
            "original policy refers to unknown historical experts",
        )
        _require(
            losses
            == library.calibration["clusters"][str(cluster.cluster_id)][
                "weighting_losses"
            ],
            "original calibrated active policy changed",
        )
        expected_losses = (
            losses
            if expert is None
            else {expert: _singleton_loss(library, cluster.cluster_id, expert)}
        )
        _require(
            cluster.weighting_losses == expected_losses,
            "promoted policy active experts or historical losses changed",
        )


def _promotion_timestamp(value: Any) -> pd.Timestamp:
    _require(
        isinstance(value, str) and bool(value.strip()),
        "selected promotion policy requires its last observed timestamp",
    )
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("promotion availability timestamp is invalid") from exc
    _require(
        not pd.isna(result), "promotion availability timestamp must not be missing"
    )
    return result


def validate_promotion_query(library: AnchorLibrary, timestamps: Any) -> None:
    """A selected policy is available only after its model-selection labels."""
    if not library.promotion or library.promotion.get("status") != "selected":
        return
    cutoff = _promotion_timestamp(library.promotion.get("promotion_last_timestamp"))
    _require(
        timestamps is not None,
        "promoted policy requires query timestamps to verify its availability",
    )
    raw = np.asarray(timestamps)
    _require(
        raw.shape == (library.lookback,),
        "promoted query timestamps must cover its complete history",
    )
    is_datetime_array = np.issubdtype(raw.dtype, np.datetime64)
    _require(
        is_datetime_array
        or not any(
            isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating))
            for value in raw
        ),
        "promoted query timestamps must be explicit dates, not numeric positions",
    )
    try:
        dates = (
            pd.DatetimeIndex(raw)
            if is_datetime_array
            else pd.DatetimeIndex(
                pd.to_datetime(list(raw), errors="raise", format="mixed")
            )
        )
        _require(
            not dates.isna().any()
            and not dates.has_duplicates
            and dates.is_monotonic_increasing,
            "promoted query timestamps must be present, unique, and increasing",
        )
        available = dates[-1] >= cutoff
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "promoted query and availability timestamps must be valid and use compatible time zones"
        ) from exc
    _require(bool(available), "promoted policy was selected after this forecast origin")
