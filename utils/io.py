from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.schema import TimeSeriesWindow


TIMESTAMP_CANDIDATES = ("date", "time_stamp", "timestamp", "ds", "time")


@dataclass(slots=True)
class DatasetView:
    frame: pd.DataFrame
    target_col: str
    timestamp_col: str | None


def load_dataset(
    path: str | Path,
    target_col: str | None = None,
    timestamp_col: str | None = None,
) -> DatasetView:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty dataset: {path}")

    ts_col = timestamp_col or infer_timestamp_column(frame)
    if ts_col and ts_col in frame.columns:
        frame[ts_col] = pd.to_datetime(frame[ts_col], errors="coerce")

    tgt_col = target_col or infer_target_column(frame, ts_col)
    frame[tgt_col] = pd.to_numeric(frame[tgt_col], errors="coerce")
    frame = frame.dropna(subset=[tgt_col]).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("dataset must contain at least two valid target values")

    return DatasetView(frame=frame, target_col=tgt_col, timestamp_col=ts_col)


def infer_timestamp_column(frame: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_CANDIDATES:
        if col in frame.columns:
            return col
    return None


def infer_target_column(frame: pd.DataFrame, timestamp_col: str | None = None) -> str:
    excluded = {timestamp_col, "predicted_ans", "features_used", "answer", "ground_truth"}
    numeric_candidates: list[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() > 0:
            numeric_candidates.append(col)
    if not numeric_candidates:
        raise ValueError("could not infer target column from numeric columns")
    return numeric_candidates[-1]


def make_windows(
    dataset: DatasetView,
    lookback: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
) -> list[TimeSeriesWindow]:
    step = stride if stride and stride > 0 else horizon
    total = len(dataset.frame)
    windows: list[TimeSeriesWindow] = []
    for start in range(0, total - lookback - horizon + 1, step):
        history = dataset.frame.iloc[start : start + lookback].copy()
        future = dataset.frame.iloc[start + lookback : start + lookback + horizon].copy()
        windows.append(
            TimeSeriesWindow(
                history=history,
                future=future,
                target_col=dataset.target_col,
                timestamp_col=dataset.timestamp_col,
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def make_latest_window(dataset: DatasetView, lookback: int) -> TimeSeriesWindow:
    if len(dataset.frame) < lookback:
        raise ValueError(f"dataset length {len(dataset.frame)} is smaller than lookback {lookback}")
    return TimeSeriesWindow(
        history=dataset.frame.iloc[-lookback:].copy(),
        future=None,
        target_col=dataset.target_col,
        timestamp_col=dataset.timestamp_col,
    )


def write_forecast_csv(
    path: str | Path,
    forecast: list[float],
    start_timestamp: pd.Timestamp | None = None,
    frequency: str | None = None,
) -> None:
    output = pd.DataFrame({"step": np.arange(1, len(forecast) + 1), "forecast": forecast})
    if start_timestamp is not None and frequency:
        output.insert(0, "time_stamp", pd.date_range(start=start_timestamp, periods=len(forecast), freq=frequency))
    output.to_csv(path, index=False)
