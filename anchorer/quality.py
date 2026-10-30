"""Chronological confirmation and explicit quality decisions for prepared priors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def representative_origins(
    start: int,
    stop: int,
    horizon: int,
    stride: int,
    limit: int,
    *,
    grid_origin: int | None = None,
) -> list[int]:
    """Choose spread-out origins whose complete target intervals stay in one split."""
    for name, value, minimum in (
        ("start", start, 0),
        ("stop", stop, 0),
        ("horizon", horizon, 1),
        ("stride", stride, 1),
        ("limit", limit, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if grid_origin is not None and (
        isinstance(grid_origin, bool)
        or not isinstance(grid_origin, int)
        or grid_origin < 0
    ):
        raise ValueError("grid_origin must be a nonnegative integer or None")
    # Nonoverlapping scoring windows remain a subset of the original stride grid.
    step = stride * ((horizon + stride - 1) // stride)
    first = (
        start
        if grid_origin is None
        else grid_origin + max(0, (start - grid_origin + step - 1) // step) * step
    )
    origins = list(range(first, stop - horizon + 1, step))
    if len(origins) <= limit:
        return origins
    indices = np.rint(np.linspace(0, len(origins) - 1, limit)).astype(int)
    return [origins[int(index)] for index in indices]


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """A recipe is eligible only when every preselected origin succeeds."""
    valid = [record for record in records if record.get("status") == "valid"]
    complete = bool(records) and len(valid) == len(records)
    return {
        "status": "valid" if complete else "failed",
        "windows": len(records),
        "valid_windows": len(valid),
        "failed_windows": len(records) - len(valid),
        "mse": (
            math.fsum(record["mse"] / len(records) for record in valid)
            if complete
            else None
        ),
        "mae": (
            math.fsum(record["mae"] / len(records) for record in valid)
            if complete
            else None
        ),
    }


def assess_quality(
    candidate: dict[str, Any],
    seasonal: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    minimum_windows: int,
    min_mse_improvement: float,
    max_mae_regression: float,
) -> dict[str, Any]:
    """Assess the frozen prior once, allowing numerical ties at the zero-error floor."""

    def valid(metrics):
        return metrics.get("status") == "valid" and all(
            not isinstance(metrics.get(key), bool)
            and isinstance(metrics.get(key), (int, float))
            and math.isfinite(metrics[key])
            and metrics[key] >= 0
            for key in ("mse", "mae")
        )

    def at_most(actual, bound):
        tolerance = 1e-10 * max(1.0, abs(actual), abs(bound))
        return actual <= bound + tolerance

    usable = all(valid(metrics) for metrics in (candidate, seasonal, incumbent))
    count = candidate.get("windows", 0)
    checks = {
        "complete_confirmation_predictions": usable,
        "minimum_confirmation_windows": count >= minimum_windows,
        "mse_improvement_over_seasonal_naive": usable
        and at_most(candidate["mse"], seasonal["mse"] * (1 - min_mse_improvement)),
        "mse_no_worse_than_default_incumbent": usable
        and at_most(candidate["mse"], incumbent["mse"]),
        "mae_no_worse_than_seasonal_naive": usable
        and at_most(candidate["mae"], seasonal["mae"] * (1 + max_mae_regression)),
        "mae_no_worse_than_default_incumbent": usable
        and at_most(candidate["mae"], incumbent["mae"] * (1 + max_mae_regression)),
    }
    return {
        "quality_passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "requirements": {
            "min_confirmation_windows": minimum_windows,
            "min_mse_improvement": min_mse_improvement,
            "max_mae_regression": max_mae_regression,
            "numerical_tolerance": "1e-10 * max(1, abs(actual), abs(bound))",
            "zero_error_floor": "ties within numerical tolerance are allowed",
        },
    }
