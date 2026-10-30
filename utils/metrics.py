from __future__ import annotations

import numpy as np


def _aligned_finite(
    prediction: list[float] | np.ndarray, truth: list[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pred, actual = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: prediction {pred.shape}, truth {actual.shape}"
        )
    if (
        not pred.size
        or pred.ndim == 0
        or not np.isfinite(pred).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("metrics require nonempty finite sequences")
    return pred, actual


def mse(prediction: list[float] | np.ndarray, truth: list[float] | np.ndarray) -> float:
    pred, actual = _aligned_finite(prediction, truth)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean((pred - actual) ** 2))
    if not np.isfinite(value):
        raise ValueError("squared error exceeds finite numerical range")
    return value


def mae(prediction: list[float] | np.ndarray, truth: list[float] | np.ndarray) -> float:
    pred, actual = _aligned_finite(prediction, truth)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean(np.abs(pred - actual)))
    if not np.isfinite(value):
        raise ValueError("absolute error exceeds finite numerical range")
    return value
