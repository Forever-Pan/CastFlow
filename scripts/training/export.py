from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def export_memory_training_csv(memory_path: str | Path, output_path: str | Path) -> str:
    """Export CastFlow memory to the old grok.csv training schema."""
    payload = json.loads(Path(memory_path).read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else payload.get("entries", [])
    rows: list[dict[str, Any]] = []
    for entry in entries:
        response = str(entry.get("Forecasting_answer") or "")
        reasoning, answer = extract_reasoning_and_answer(response)
        rows.append(
            {
                "idx": entry.get("idx", len(rows)),
                "dataset_name": entry.get("dataset_name", ""),
                "input": entry.get("input_json", ""),
                "ground_truth": entry.get("ground_truth", ""),
                "tool": json.dumps(entry.get("tool_calls", []), ensure_ascii=False),
                "prompt": entry.get("forecast_full_prompt", ""),
                "response": response,
                "reasoning": reasoning,
                "answer": answer,
            }
        )
    rows.sort(key=lambda row: (str(row.get("dataset_name", "")), int(row["idx"]) if str(row["idx"]).isdigit() else 0))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["idx", "dataset_name", "input", "ground_truth", "tool", "prompt", "response", "reasoning", "answer"]).to_csv(output, index=False)
    return str(output)


def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    reasoning = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return reasoning, answer
