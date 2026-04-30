from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class MemoryMatch:
    entry: dict[str, Any]
    similarity: float


class StrategyMemory:
    """Strategy memory storing distilled successful CastFlow trajectories."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(
        self,
        target_history: np.ndarray,
        tool_schedule: list[str],
        diagnostic_outputs: dict[str, Any],
        trajectory: str,
        metrics: dict[str, float] | None = None,
        input_json: str = "",
        ground_truth: str = "",
        forecast_full_prompt: str = "",
        forecast_answer: str = "",
        model_name: str = "",
        dataset_name: str | None = None,
    ) -> None:
        idx = int(metrics.get("window_idx", len(self.entries))) if metrics else len(self.entries)
        tool_outputs = legacy_tool_outputs(diagnostic_outputs)
        self.entries.append(
            {
                "idx": idx,
                "dataset_name": dataset_name or "",
                "input_json": input_json,
                "ground_truth": ground_truth,
                "tool_calls": legacy_tool_calls(tool_schedule, tool_outputs),
                "tool_outputs": tool_outputs,
                "forecast_full_prompt": forecast_full_prompt,
                "Forecasting_answer": forecast_answer,
                "model_name": model_name,
                "metrics": metrics or {},
                "vector": embed_time_series(target_history).tolist(),
            }
        )

    def retrieve(
        self,
        target_history: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float = 0.80,
    ) -> list[MemoryMatch]:
        if not self.entries:
            return []
        query = embed_time_series(target_history)
        matches: list[MemoryMatch] = []
        for entry in self.entries:
            vector = entry.get("embedding", entry.get("vector"))
            if vector is None:
                continue
            sim = cosine_similarity(query, np.asarray(vector, dtype=float))
            if sim >= similarity_threshold:
                matches.append(MemoryMatch(entry=entry, similarity=sim))
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StrategyMemory":
        memory = cls()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            memory.entries = payload
        elif isinstance(payload, dict):
            memory.entries = payload.get("entries", [])
        else:
            memory.entries = []
        for entry in memory.entries:
            if isinstance(entry, dict) and entry.get("vector") is None and entry.get("embedding") is None:
                rebuilt = vector_from_input_json(entry.get("input_json"))
                if rebuilt is not None:
                    entry["vector"] = rebuilt.tolist()
        return memory


LEGACY_TOOL_NAME_MAP = {
    "statistical_analysis_tool": "statistical_analysis",
    "trend_analysis_tool": "trend_analysis",
    "exogenous_analysis_tool": "exogenous_analysis",
    "autoregressive_residual_tool": "ar_residual_tool",
}


def legacy_tool_outputs(diagnostic_outputs: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name, output in diagnostic_outputs.items():
        legacy_name = LEGACY_TOOL_NAME_MAP.get(name, name)
        if name == "model_auxiliary_tool" and isinstance(output, dict):
            outputs[legacy_name] = {
                "best_model": output.get("best_model"),
                "reference_prediction": output.get("reference_prediction") or output.get("ensemble_forecast_baseline"),
                "forecast_length": len(output.get("reference_prediction") or output.get("ensemble_forecast_baseline") or []),
            }
        else:
            outputs[legacy_name] = output
    return outputs


def legacy_tool_calls(tool_schedule: list[str], tool_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for name in tool_schedule:
        legacy_name = LEGACY_TOOL_NAME_MAP.get(name, name)
        if legacy_name not in tool_outputs:
            continue
        calls.append(
            {
                "tool_name": legacy_name,
                "params": {"input_data": "provided"},
                "output": tool_outputs[legacy_name],
            }
        )
    return calls


def embed_time_series(values: np.ndarray, n_samples: int = 24) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return np.zeros(5 + n_samples, dtype=np.float32)

    x = np.arange(series.size, dtype=float)
    slope = float(np.polyfit(x, series, 1)[0]) if series.size > 1 else 0.0
    stats = np.array(
        [
            float(np.mean(series)),
            float(np.std(series)),
            float(np.min(series)),
            float(np.max(series)),
            slope,
        ],
        dtype=float,
    )
    if series.size >= n_samples:
        indices = np.linspace(0, series.size - 1, n_samples, dtype=int)
        sampled = series[indices].astype(float)
    elif series.size == 1:
        sampled = np.repeat(series[0], n_samples)
    else:
        idx = np.linspace(0, series.size - 1, n_samples)
        sampled = np.interp(idx, np.arange(series.size), series)
    return np.asarray(np.concatenate([stats, sampled.astype(float)]), dtype=np.float32)


def vector_from_input_json(input_json: object) -> np.ndarray | None:
    try:
        payload = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(payload, dict):
            return None
        values = np.asarray(list(payload.values()), dtype=float)
    except Exception:
        return None
    return embed_time_series(values)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)
