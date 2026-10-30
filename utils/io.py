from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os
import tempfile
from typing import Any

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
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next((row for row in csv.reader(handle) if row), [])
    if len(header) != len(set(header)):
        raise ValueError("dataset contains duplicate column names")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty dataset: {path}")

    ts_col = timestamp_col or infer_timestamp_column(frame)
    if timestamp_col and timestamp_col not in frame:
        raise ValueError(f"timestamp column {timestamp_col!r} not found")
    if ts_col:
        frame[ts_col] = pd.to_datetime(frame[ts_col], errors="raise", format="mixed")
        timestamps = frame[ts_col]
        if (
            timestamps.isna().any()
            or timestamps.duplicated().any()
            or not timestamps.is_monotonic_increasing
        ):
            raise ValueError(
                "timestamps must be present, unique, and strictly increasing"
            )

    tgt_col = target_col or infer_target_column(frame, ts_col)
    if tgt_col not in frame or tgt_col == ts_col:
        raise ValueError(
            "target column must exist and differ from the timestamp column"
        )
    frame[tgt_col] = pd.to_numeric(frame[tgt_col], errors="coerce")
    if not np.isfinite(frame[tgt_col].to_numpy(dtype=float)).all():
        raise ValueError(
            "target values must be finite; preprocess missing values without deleting time steps"
        )
    if len(frame) < 2:
        raise ValueError("dataset must contain at least two valid target values")

    return DatasetView(frame=frame, target_col=tgt_col, timestamp_col=ts_col)


def infer_timestamp_column(frame: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_CANDIDATES:
        if col in frame.columns:
            return col
    return None


def infer_target_column(frame: pd.DataFrame, timestamp_col: str | None = None) -> str:
    excluded = {
        timestamp_col,
        "predicted_ans",
        "features_used",
        "answer",
        "ground_truth",
    }
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
    for name, value in (("lookback", lookback), ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if stride is not None and (
        isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0
    ):
        raise ValueError("stride must be a positive integer")
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows <= 0
    ):
        raise ValueError("max_windows must be a positive integer")
    step = horizon if stride is None else stride
    total = len(dataset.frame)
    windows: list[TimeSeriesWindow] = []
    for start in range(0, total - lookback - horizon + 1, step):
        history = dataset.frame.iloc[start : start + lookback].copy()
        future = dataset.frame.iloc[
            start + lookback : start + lookback + horizon
        ].copy()
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
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    if len(dataset.frame) < lookback:
        raise ValueError(
            f"dataset length {len(dataset.frame)} is smaller than lookback {lookback}"
        )
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
    output = pd.DataFrame(
        {"step": np.arange(1, len(forecast) + 1), "forecast": forecast}
    )
    if start_timestamp is not None and frequency:
        output.insert(
            0,
            "time_stamp",
            pd.date_range(start=start_timestamp, periods=len(forecast), freq=frequency),
        )
    if not forecast or not np.isfinite(np.asarray(forecast, dtype=float)).all():
        raise ValueError("forecast CSV requires a nonempty finite forecast")
    atomic_write_csv(path, output)


def forecast_timestamps(window: TimeSeriesWindow, horizon: int) -> list[str]:
    """Return the known horizon schedule without reading target values."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    name = window.timestamp_col
    if not name:
        return []
    schedules: list[list[pd.Timestamp]] = []
    for frame in (window.future, window.covariate_forecasts):
        if frame is not None and name in frame:
            if len(frame) != horizon:
                raise ValueError("forecast schedule length differs from the horizon")
            values = pd.to_datetime(frame[name], errors="raise")
            if (
                values.isna().any()
                or values.duplicated().any()
                or not values.is_monotonic_increasing
            ):
                raise ValueError(
                    "forecast timestamps must be unique and strictly increasing"
                )
            if (
                window.forecast_origin is not None
                and not (values > window.forecast_origin).all()
            ):
                raise ValueError(
                    "forecast timestamps must be after the forecast origin"
                )
            schedules.append([pd.Timestamp(v) for v in values])
    if schedules:
        if any(schedule != schedules[0] for schedule in schedules[1:]):
            raise ValueError(
                "auxiliary forecast timestamps differ from the requested evaluation horizon"
            )
        return [value.isoformat(sep=" ") for value in schedules[0]]
    history = pd.DatetimeIndex(pd.to_datetime(window.history[name], errors="raise"))
    if (
        len(history) < 2
        or history.hasnans
        or not history.is_monotonic_increasing
        or history.has_duplicates
    ):
        return []
    freq = pd.infer_freq(history) if len(history) >= 3 else None
    if freq:
        future = pd.date_range(start=history[-1], periods=horizon + 1, freq=freq)[1:]
    else:
        diffs = history[1:] - history[:-1]
        if not (diffs == diffs[0]).all() or diffs[0] <= pd.Timedelta(0):
            return []
        future = pd.date_range(
            start=history[-1] + diffs[0], periods=horizon, freq=diffs[0]
        )
    return [pd.Timestamp(v).isoformat(sep=" ") for v in future]


def attach_covariate_forecasts(
    window: TimeSeriesWindow,
    forecasts: pd.DataFrame,
    horizon: int,
    *,
    issued_at: str | pd.Timestamp | None = None,
    issue_col: str = "issued_at",
) -> None:
    """Select the latest origin-admissible forecast for each horizon timestamp.

    The input is a separately supplied ex ante forecast table, never the held-out
    observation frame. A single issue time or per-row issue times are required.
    """
    timestamp = window.timestamp_col
    origin = window.forecast_origin
    if not timestamp or origin is None or timestamp not in forecasts:
        raise ValueError(
            "auxiliary forecasts require an explicit timestamp column and forecast origin"
        )
    if window.target_col in forecasts:
        raise ValueError(
            "auxiliary forecast inputs must not include the future target column"
        )
    frame = forecasts.copy()
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise", format="mixed")
    if issue_col in frame:
        issue = pd.to_datetime(frame[issue_col], errors="raise", format="mixed")
    elif issued_at is not None:
        issue = pd.Series(pd.Timestamp(issued_at), index=frame.index)
    else:
        raise ValueError("provide issued_at for auxiliary forecast inputs")
    if frame[timestamp].isna().any() or issue.isna().any():
        raise ValueError("auxiliary timestamps and issue times must be present")
    try:
        eligible = (issue <= origin) & (frame[timestamp] > origin)
    except TypeError as exc:
        raise ValueError(
            "history, horizon, and issue timestamps must use compatible time zones"
        ) from exc
    frame["__issue_time__"] = issue
    schedule = forecast_timestamps(window, horizon)
    usable = frame.loc[eligible].sort_values([timestamp, "__issue_time__"])
    if usable.duplicated([timestamp, "__issue_time__"]).any():
        raise ValueError(
            "duplicate auxiliary forecasts for the same timestamp and issue time"
        )
    usable = usable.drop_duplicates(timestamp, keep="last")
    if schedule:
        expected = pd.DatetimeIndex(pd.to_datetime(schedule))
        usable = usable.set_index(timestamp).reindex(expected)
        if usable["__issue_time__"].isna().any():
            raise ValueError(
                "origin-admissible auxiliary forecasts do not cover the requested horizon"
            )
        usable.index.name = timestamp
        usable = usable.reset_index()
    else:
        usable = usable.iloc[:horizon]
    if len(usable) != horizon:
        raise ValueError("auxiliary forecasts must cover every requested horizon step")
    feature_cols = [
        c for c in usable if c not in {timestamp, issue_col, "__issue_time__"}
    ]
    if not feature_cols:
        raise ValueError("auxiliary forecast table contains no covariates")
    for col in feature_cols:
        usable[col] = pd.to_numeric(usable[col], errors="raise")
        if not np.isfinite(usable[col].to_numpy(dtype=float)).all():
            raise ValueError(
                f"auxiliary forecast column {col!r} must contain finite values"
            )
    window.covariate_issued_at = pd.Timestamp(usable["__issue_time__"].max())
    window.covariate_forecasts = usable[[timestamp, *feature_cols]].reset_index(
        drop=True
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _atomic_write(path: str | Path, writer: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(
                target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    _atomic_write(
        path,
        lambda handle: json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        ),
    )


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    _atomic_write(path, lambda handle: frame.to_csv(handle, index=False))
