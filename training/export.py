from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluate import (
    aligned_arrays,
    extract_answer_text,
    parse_forecast_value,
)
from training.rewards import (
    build_small_model_prediction_dict,
    canonical_dataset_name,
    serialize_forecast,
)
from utils.io import atomic_write_csv


def export_memory_training_csv(memory_path: str | Path, output_path: str | Path) -> str:
    """Export selected teacher continuations and their fixed evidence contexts."""
    if Path(memory_path).resolve() == Path(output_path).resolve():
        raise ValueError("training output must differ from the memory input")
    payload = json.loads(Path(memory_path).read_text(encoding="utf-8"))
    entries = (
        payload
        if isinstance(payload, list)
        else payload.get("entries") if isinstance(payload, dict) else None
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError("memory contains no training trajectories")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"memory entry {index} is not an object")
        response = entry.get("Forecasting_answer", entry.get("forecast_answer", ""))
        prompt = entry.get("forecast_full_prompt", "")
        system_prompt = entry.get("forecast_system_prompt", "")
        if system_prompt is None:
            system_prompt = ""
        if not isinstance(system_prompt, str):
            raise ValueError(f"memory entry {index} has a non-text system prompt")
        if (
            not isinstance(response, str)
            or not response.strip()
            or not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise ValueError(
                f"memory entry {index} lacks its teacher response or forecast context"
            )
        truth, forecast = parse_forecast_value(
            entry.get("ground_truth")
        ), parse_forecast_value(response)
        try:
            aligned_arrays(forecast, truth)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"memory entry {index} has an invalid or unaligned teacher forecast/ground_truth"
            ) from exc
        raw_name = entry.get("dataset_name", "")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"memory entry {index} requires its dataset name")
        try:
            name = canonical_dataset_name(raw_name)
        except ValueError:
            name = raw_name.strip()
        tool = entry.get("tool_calls", entry.get("tool_outputs", []))
        baseline = build_small_model_prediction_dict(tool, truth)
        reasoning, answer = extract_reasoning_and_answer(response)
        rows.append(
            {
                "idx": entry.get("idx", len(rows)),
                "dataset_name": name,
                "input": entry.get("input_json", ""),
                "ground_truth": serialize_forecast(truth),
                "tool": serialize_forecast(tool),
                "baseline_forecast": serialize_forecast(baseline) if baseline else "",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": response,
                "reasoning": reasoning,
                "answer": answer,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("dataset_name", "")),
            int(row["idx"]) if str(row["idx"]).isdigit() else 0,
        )
    )
    output = Path(output_path)
    atomic_write_csv(
        output,
        pd.DataFrame(
            rows,
            columns=[
                "idx",
                "dataset_name",
                "input",
                "ground_truth",
                "tool",
                "baseline_forecast",
                "prompt",
                "system_prompt",
                "response",
                "reasoning",
                "answer",
            ],
        ),
    )
    return str(output)


def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    think_match = re.search(
        r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE
    )
    reasoning = think_match.group(1).strip() if think_match else ""
    answer = extract_answer_text(text)
    return reasoning, answer
