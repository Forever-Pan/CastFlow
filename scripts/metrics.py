from __future__ import annotations

import numpy as np


def mse(prediction: list[float] | np.ndarray, truth: list[float] | np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, truth {actual.shape}")
    return float(np.mean((pred - actual) ** 2))


def mae(prediction: list[float] | np.ndarray, truth: list[float] | np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape}, truth {actual.shape}")
    return float(np.mean(np.abs(pred - actual)))
