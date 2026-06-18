from __future__ import annotations

import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluate import compute_metrics, parse_forecast_value


@dataclass(slots=True)
class InferenceConfig:
    base_url: str = "http://localhost:8003/v1"
    model: str = "forecast"
    api_key: str = "test-key"
    workers: int = 16
    timeout: float = 600.0
    max_tokens: int | None = 5000
    temperature: float = 0.3
    max_retries: int = 3
    prompt_col: str = "prompt"
    output_col: str = "answer_local_LLM"
    resume: bool = True


def run_openai_compatible_inference(
    input_path: str | Path,
    output_path: str | Path,
    config: InferenceConfig,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI-compatible inference requires the openai package") from exc

    frame = pd.read_csv(input_path, keep_default_na=False)
    if config.prompt_col not in frame.columns:
        raise ValueError(f"prompt column {config.prompt_col!r} not found; columns={list(frame.columns)}")
    if "idx" not in frame.columns:
        frame["idx"] = frame.index
    has_ground_truth = "ground_truth" in frame.columns

    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)
    existing_results = load_existing_valid_results(output_path, frame, config.output_col) if config.resume else {}
    rows_to_process = frame[~frame["idx"].isin(existing_results)].copy()

    def call(row: pd.Series) -> tuple[int, dict[str, Any]]:
        idx = int(row["idx"])
        prompt = str(row[config.prompt_col])
        last_error = ""
        for attempt in range(1, config.max_retries + 1):
            started = time.time()
            try:
                params: dict[str, Any] = {
                    "model": config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.temperature,
                }
                if config.max_tokens:
                    params["max_tokens"] = config.max_tokens
                response = client.chat.completions.create(**params)
                text = (response.choices[0].message.content or "").strip()
                reasoning, answer = parse_response(text)
                if has_ground_truth:
                    ground_truth = str(row.get("ground_truth", "")).strip()
                    if not is_valid_answer(answer, ground_truth):
                        last_error = "Invalid answer (cannot calculate MSE)"
                        if attempt < config.max_retries:
                            continue
                        return idx, result_payload(config, answer, text, reasoning, False, last_error, started, attempt)
                return idx, result_payload(config, answer, text, reasoning, True, "", started, attempt)
            except Exception as exc:  # noqa: BLE001 - preserve API error text for batch diagnosis
                last_error = f"{type(exc).__name__}: {exc}"
        return idx, {
            config.output_col: "",
            "full_response": "",
            "reasoning": "",
            "success": False,
            "error": last_error,
            "elapsed": 0.0,
            "attempt": config.max_retries,
        }

    results: dict[int, dict[str, Any]] = {
        int(idx): {
            config.output_col: answer,
            "full_response": "",
            "reasoning": "",
            "success": True,
            "error": "",
            "elapsed": 0.0,
            "attempt": 0,
        }
        for idx, answer in existing_results.items()
    }
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(call, row) for _, row in rows_to_process.iterrows()]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    for col in [config.output_col, "full_response", "reasoning", "success", "error", "elapsed", "attempt"]:
        frame[col] = [results.get(int(idx), {}).get(col, "") for idx in frame["idx"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def result_payload(
    config: InferenceConfig,
    answer: str,
    full_response: str,
    reasoning: str,
    success: bool,
    error: str,
    started: float,
    attempt: int,
) -> dict[str, Any]:
    return {
        config.output_col: answer,
        "full_response": full_response,
        "reasoning": reasoning,
        "success": success,
        "error": error,
        "elapsed": round(time.time() - started, 4),
        "attempt": attempt,
    }


def parse_response(response_text: str) -> tuple[str, str]:
    reasoning = ""
    answer = ""
    if not response_text:
        return reasoning, answer

    reasoning_match = re.search(r"<think>(.*?)</think>", response_text, flags=re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    if not answer:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if code_block and re.search(r'"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"', code_block.group(1)):
            answer = code_block.group(1).strip()
    if not answer:
        answer = extract_json_object_with_timestamps(response_text)
    return reasoning, answer


def extract_json_object_with_timestamps(text: str) -> str:
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    if re.search(r'"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"', candidate):
                        return candidate
                    break
        start = text.find("{", start + 1)
    return ""


def is_valid_answer(answer: str, ground_truth: str) -> bool:
    if not answer.strip() or not ground_truth.strip():
        return False
    metrics = compute_metrics(parse_forecast_value(answer), parse_forecast_value(ground_truth))
    return not pd.isna(metrics["mse"])


def load_existing_valid_results(
    output_path: str | Path,
    input_frame: pd.DataFrame,
    output_col: str,
) -> dict[int, str]:
    path = Path(output_path)
    if not path.exists():
        return {}
    try:
        existing = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return {}
    if "idx" not in existing.columns or output_col not in existing.columns:
        return {}
    if "ground_truth" not in existing.columns and "ground_truth" in input_frame.columns:
        existing = existing.merge(input_frame[["idx", "ground_truth"]], on="idx", how="left")

    has_ground_truth = "ground_truth" in existing.columns
    results: dict[int, str] = {}
    for _, row in existing.iterrows():
        answer = str(row.get(output_col, "")).strip()
        if not answer or answer.lower() == "nan":
            continue
        if has_ground_truth and not is_valid_answer(answer, str(row.get("ground_truth", "")).strip()):
            continue
        try:
            results[int(row["idx"])] = answer
        except (TypeError, ValueError):
            continue
    return results
