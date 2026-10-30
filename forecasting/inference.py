from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.evaluate import (
    aligned_arrays,
    extract_answer_text,
    parse_forecast_value,
)
from utils.io import atomic_write_csv


@dataclass(slots=True)
class InferenceConfig:
    base_url: str = "http://localhost:8003/v1"
    model: str = "forecast"
    api_key: str = "EMPTY"
    workers: int = 16
    timeout: float = 600.0
    max_tokens: int | None = 5000
    temperature: float = 0.3
    max_retries: int = 3
    prompt_col: str = "prompt"
    output_col: str = "answer_local_LLM"
    resume: bool = True

    def validate(self) -> None:
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError("workers must be a positive integer")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 1
        ):
            raise ValueError("max_retries must be a positive total attempt count")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer or None")
        if (
            not self.model
            or not self.base_url
            or not self.prompt_col
            or not self.output_col
        ):
            raise ValueError("endpoint, model and column names must be nonempty")
        if self.output_col in {
            self.prompt_col,
            "idx",
            "ground_truth",
            "horizon",
            "_castflow_input_hash",
            "full_response",
            "reasoning",
            "success",
            "error",
            "elapsed",
            "attempt",
        }:
            raise ValueError(
                "output_col must not overwrite input identifiers or result metadata"
            )


def input_fingerprint(row: pd.Series, config: InferenceConfig) -> str:
    payload = {
        "prompt": str(row[config.prompt_col]),
        "model": config.model,
        "endpoint": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "horizon": str(row.get("horizon", "")),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def run_openai_compatible_inference(
    input_path: str | Path, output_path: str | Path, config: InferenceConfig
) -> str:
    config.validate()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI-compatible inference requires the openai package"
        ) from exc
    frame = pd.read_csv(input_path, keep_default_na=False)
    if frame.empty or config.prompt_col not in frame:
        raise ValueError(
            "input must contain at least one row and the configured prompt column"
        )
    if "idx" not in frame:
        frame["idx"] = frame.index
    numeric = pd.to_numeric(frame["idx"], errors="raise")
    if (
        not all(math.isfinite(value) and float(value).is_integer() for value in numeric)
        or numeric.duplicated().any()
    ):
        raise ValueError("idx must contain unique finite integer identifiers")
    frame["idx"] = numeric.astype("int64")
    for _, row in frame.iterrows():
        horizon = requested_horizon(row)
        target = str(row.get("ground_truth", "")).strip()
        if target:
            values = parse_forecast_value(target)
            if not values or (horizon is not None and len(values) != horizon):
                raise ValueError(
                    "ground_truth must contain a complete finite horizon when supplied"
                )
    frame["_castflow_input_hash"] = [
        input_fingerprint(row, config) for _, row in frame.iterrows()
    ]
    results = (
        _load_existing_payloads(output_path, frame, config) if config.resume else {}
    )
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=0,
    )

    def call(row: pd.Series) -> tuple[int, dict[str, Any]]:
        idx = int(row["idx"])
        started = time.perf_counter()
        last_error = ""
        answer = full = reasoning = ""
        for attempt in range(1, config.max_retries + 1):
            try:
                params = {
                    "model": config.model,
                    "messages": [
                        {"role": "user", "content": str(row[config.prompt_col])}
                    ],
                    "temperature": config.temperature,
                }
                if config.max_tokens is not None:
                    params["max_tokens"] = config.max_tokens
                response = client.chat.completions.create(**params)
                if not response.choices:
                    raise ValueError("model returned no response choices")
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                separate_reasoning = (
                    getattr(choice.message, "reasoning_content", None) or ""
                )
                full = (
                    f"<think>{separate_reasoning}</think>\n{content}"
                    if separate_reasoning
                    and not re.search(r"<think\b", content, re.IGNORECASE)
                    else content
                )
                reasoning, answer = parse_response(full)
                valid = answer_matches_row(answer, row)
                if getattr(choice, "finish_reason", None) in {
                    "length",
                    "content_filter",
                }:
                    valid = False
                if valid:
                    return idx, result_payload(
                        config, answer, full, reasoning, True, "", started, attempt
                    )
                last_error = "response is absent, malformed, truncated, or not aligned with the requested horizon"
            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc}".replace(config.api_key, "[redacted]")
                    if config.api_key
                    else type(exc).__name__
                )
                status = getattr(exc, "status_code", None)
                if (
                    status is not None
                    and status not in {408, 409, 425, 429}
                    and status < 500
                ):
                    break
            if attempt < config.max_retries:
                time.sleep(min(0.25 * 2 ** (attempt - 1), 2.0))
        return idx, result_payload(
            config, answer, full, reasoning, False, last_error, started, attempt
        )

    columns = [
        config.output_col,
        "full_response",
        "reasoning",
        "success",
        "error",
        "elapsed",
        "attempt",
    ]

    def checkpoint() -> None:
        output = frame.copy()
        for column in columns:
            output[column] = [
                results.get(int(idx), {}).get(column, "") for idx in output["idx"]
            ]
        atomic_write_csv(output_path, output)

    try:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(call, row)
                for _, row in frame.iterrows()
                if int(row["idx"]) not in results
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                checkpoint()
        checkpoint()
    finally:
        client.close()
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
        config.output_col: answer if success else "",
        "full_response": full_response,
        "reasoning": reasoning,
        "success": success,
        "error": error,
        "elapsed": round(time.perf_counter() - started, 4),
        "attempt": attempt,
    }


def parse_response(response_text: str) -> tuple[str, str]:
    if not isinstance(response_text, str) or not response_text.strip():
        return "", ""
    match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
    reasoning = match.group(1).strip() if match else ""
    answer = extract_answer_text(response_text)
    return reasoning, answer if parse_forecast_value(answer) else ""


def extract_json_object_with_timestamps(text: str) -> str:
    answer = extract_answer_text(text)
    parsed = parse_forecast_value(answer)
    if parsed and all(re.match(r"^\d{4}-\d{2}-\d{2}", str(key)) for key in parsed):
        return answer
    return ""


def is_valid_answer(answer: str, ground_truth: str) -> bool:
    """Check schema and timestamp alignment without consulting prediction error."""
    prediction = parse_forecast_value(answer)
    target = parse_forecast_value(ground_truth)
    if not prediction or not target:
        return False
    try:
        aligned_arrays(prediction, target)
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def answer_matches_row(answer: str, row: pd.Series) -> bool:
    prediction = parse_forecast_value(answer)
    if not prediction:
        return False
    horizon = requested_horizon(row)
    if horizon is not None and len(prediction) != horizon:
        return False
    if str(row.get("ground_truth", "")).strip():
        return is_valid_answer(answer, str(row["ground_truth"]))
    return True


def requested_horizon(row: pd.Series) -> int | None:
    raw = row.get("horizon", "")
    if not str(raw).strip():
        return None
    value = float(raw)
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise ValueError("row horizon must be a positive integer")
    return int(value)


def _load_existing_payloads(
    output_path: str | Path, input_frame: pd.DataFrame, config: InferenceConfig
) -> dict[int, dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return {}
    existing = pd.read_csv(path, keep_default_na=False)
    required = {"idx", config.output_col, "_castflow_input_hash", "success"}
    if not required.issubset(existing):
        return {}
    if existing["idx"].duplicated().any():
        raise ValueError("resume file contains duplicate row identifiers")
    expected = {int(row["idx"]): row for _, row in input_frame.iterrows()}
    results = {}
    for _, row in existing.iterrows():
        idx = int(row["idx"])
        source = expected.get(idx)
        if source is None or str(row["success"]).lower() not in {"true", "1"}:
            continue
        if row["_castflow_input_hash"] != source["_castflow_input_hash"]:
            continue
        answer = str(row[config.output_col])
        if not answer_matches_row(answer, source):
            continue
        results[idx] = {
            column: row.get(column, "")
            for column in [
                config.output_col,
                "full_response",
                "reasoning",
                "success",
                "error",
                "elapsed",
                "attempt",
            ]
        }
    return results


def load_existing_valid_results(
    output_path: str | Path, input_frame: pd.DataFrame, output_col: str
) -> dict[int, str]:
    """Return answers only when saved prompts match the supplied input rows."""
    config = InferenceConfig(output_col=output_col)
    frame = input_frame.copy()
    if "idx" not in frame:
        frame["idx"] = frame.index
    if config.prompt_col not in frame:
        return {}
    frame["_castflow_input_hash"] = [
        input_fingerprint(row, config) for _, row in frame.iterrows()
    ]
    return {
        idx: row[output_col]
        for idx, row in _load_existing_payloads(output_path, frame, config).items()
    }
