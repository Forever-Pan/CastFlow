"""Training-only retention of distinct expert recipes and family defaults."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from backends.catalog import content_fingerprint


def prediction_trace(trial: dict[str, Any]) -> np.ndarray:
    records = trial.get("records", [])
    if (
        trial.get("status") != "valid"
        or not records
        or any(
            record.get("status") != "valid" or "prediction" not in record
            for record in records
        )
    ):
        raise ValueError(
            "variant retention requires a complete successful selection trace"
        )
    values = np.asarray(
        [value for record in records for value in record["prediction"]], dtype=float
    )
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("selection predictions must be finite complete vectors")
    return values


def equivalent_predictions(
    left,
    right,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    correlation: float = 0.999,
    relative_rmse: float = 0.01,
) -> bool:
    """Prune numerical clones, preserving correlated but materially shifted forecasts."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("prediction traces must have equal finite shapes")
    if np.allclose(a, b, atol=atol, rtol=rtol):
        return True
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    a, b = a / scale, b / scale
    spread = max(float(np.std(a)), float(np.std(b)), 1e-12)
    distance = math.sqrt(float(np.mean((a - b) ** 2))) / spread
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = (
        float(np.clip(np.dot(a, b) / denominator, -1, 1))
        if denominator > 1e-15
        else 0.0
    )
    return distance <= relative_rmse and corr >= correlation


def retain_family_variants(
    trials: list[dict[str, Any]],
    *,
    tuned_per_family: int = 3,
    prediction_atol: float = 1e-6,
    prediction_rtol: float = 1e-5,
    correlation_threshold: float = 0.999,
    relative_rmse_threshold: float = 0.01,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the valid original default and up to N distinct tuned variants per family.

    Trial records carry ``original_default`` and a complete selection prediction
    trace. Confirmation values are neither accepted nor consulted here.
    """
    if (
        isinstance(tuned_per_family, bool)
        or not isinstance(tuned_per_family, int)
        or tuned_per_family < 0
    ):
        raise ValueError("tuned_per_family must be a nonnegative integer")
    for name, value in (
        ("prediction_atol", prediction_atol),
        ("prediction_rtol", prediction_rtol),
        ("relative_rmse_threshold", relative_rmse_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(correlation_threshold) or not -1 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between -1 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    order = {id(trial): index for index, trial in enumerate(trials)}
    for trial in trials:
        candidate = trial["candidate"]
        family = candidate.get("family") or candidate["name"]
        if not isinstance(family, str) or not family.strip():
            raise ValueError("each expert must identify a nonempty base family")
        groups.setdefault(family, []).append(trial)
    retained, summaries, default_trials = [], [], []
    for family, group in groups.items():
        valid = [trial for trial in group if trial.get("status") == "valid"]
        defaults = [trial for trial in valid if trial.get("original_default") is True]
        if len(defaults) > 1:
            raise ValueError(f"family {family!r} has multiple original defaults")
        if not valid:
            raise ValueError(f"{family}: no recipe succeeded on every selection origin")
        ordered = sorted(
            valid, key=lambda trial: (trial["mse"], trial["mae"], order[id(trial)])
        )
        chosen = defaults[:]
        traces = [prediction_trace(trial) for trial in chosen]
        duplicates = []
        tuned = 0
        for trial in ordered:
            if defaults and trial is defaults[0]:
                continue
            trace = prediction_trace(trial)
            if any(
                equivalent_predictions(
                    trace,
                    old,
                    atol=prediction_atol,
                    rtol=prediction_rtol,
                    correlation=correlation_threshold,
                    relative_rmse=relative_rmse_threshold,
                )
                for old in traces
            ):
                duplicates.append(trial["candidate_id"])
                continue
            if tuned >= tuned_per_family:
                continue
            chosen.append(trial)
            traces.append(trace)
            tuned += 1
        if not chosen:
            raise ValueError(
                f"{family}: no valid default and no permitted tuned variant"
            )
        aliases = []
        for trial in chosen:
            spec = copy.deepcopy(trial["candidate"])
            original = bool(trial.get("original_default"))
            recipe = {
                key: spec[key]
                for key in ("name", "kind", "params", "entrypoint", "kwargs")
                if key in spec
            }
            alias = (
                spec["name"]
                if original
                else f"{family}-v{content_fingerprint(recipe)[:12]}"
            )
            spec.update(
                name=alias,
                family=family,
                variant_role="default" if original else "tuned",
                selection_candidate_id=trial["candidate_id"],
            )
            retained.append(spec)
            aliases.append(alias)
            if original:
                default_trials.append({**trial, "retained_name": alias})
        summaries.append(
            {
                "family": family,
                "attempted_trials": len(group),
                "successful_trials": len(valid),
                "failed_trials": len(group) - len(valid),
                "original_default_valid": bool(defaults),
                "retained_variants": aliases,
                "retained_count": len(chosen),
                "prediction_duplicates": duplicates,
                "selected_mse": ordered[0]["mse"],
                "selected_mae": ordered[0]["mae"],
            }
        )
    if not default_trials:
        raise ValueError(
            "no original default expert succeeded on the complete selection design"
        )
    incumbent = min(
        default_trials,
        key=lambda trial: (trial["mse"], trial["mae"], trial["retained_name"]),
    )
    names = [spec["name"] for spec in retained]
    if len(set(names)) != len(names):
        raise ValueError("retained expert aliases collide")
    policy = {
        "tuned_per_family": tuned_per_family,
        "original_default_retained_when_valid": True,
        "prediction_atol": prediction_atol,
        "prediction_rtol": prediction_rtol,
        "correlation_threshold": correlation_threshold,
        "relative_rmse_threshold": relative_rmse_threshold,
        "confirmation_used": False,
        "incumbent_name": incumbent["retained_name"],
        "incumbent_recipe": incumbent["candidate"],
    }
    return retained, summaries, policy
