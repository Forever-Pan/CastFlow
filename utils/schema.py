from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TimeSeriesWindow:
    """History, optional evaluation labels, and separately issued forecasts.

    ``future`` is reserved for offline labels, never inference evidence.
    """

    history: pd.DataFrame
    future: pd.DataFrame | None
    target_col: str
    timestamp_col: str | None
    covariate_forecasts: pd.DataFrame | None = None
    covariate_issued_at: pd.Timestamp | None = None

    @property
    def forecast_origin(self) -> pd.Timestamp | None:
        if (
            self.timestamp_col
            and self.timestamp_col in self.history
            and not self.history.empty
        ):
            return pd.Timestamp(self.history[self.timestamp_col].iloc[-1])
        return None

    @property
    def target_history(self) -> np.ndarray:
        return self.history[self.target_col].to_numpy(dtype=float)

    @property
    def target_future(self) -> np.ndarray | None:
        if self.future is None:
            return None
        return self.future[self.target_col].to_numpy(dtype=float)


@dataclass(slots=True)
class ToolObservation:
    tool_name: str
    category: str
    output: dict[str, Any]


@dataclass(slots=True)
class WorkflowState:
    """State used by the CastFlow planning-action-forecasting-reflection loop."""

    window: TimeSeriesWindow
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    tool_schedule: list[str] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    baseline_forecast: list[float] | None = None
    forecast: list[float] | None = None
    forecast_full_prompt: str = ""
    forecast_full_response: str = ""
    reflection: str = ""
    retry_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    expected_timestamps: list[str] = field(default_factory=list)
    forecast_system_prompt: str = ""

    def observation_map(self) -> dict[str, dict[str, Any]]:
        return {obs.tool_name: obs.output for obs in self.observations}


@dataclass(slots=True)
class ForecastResult:
    forecast: list[float]
    baseline_forecast: list[float] | None
    tool_schedule: list[str]
    observations: dict[str, dict[str, Any]]
    reflection: str
    forecast_full_prompt: str = ""
    forecast_full_response: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "accepted"
    failure_reason: str = ""
    attempts: int = 1
    fallback_source: str | None = None
    forecast_system_prompt: str = ""
