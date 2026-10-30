from __future__ import annotations

from pathlib import Path
import tempfile
import warnings
import shutil

import pandas as pd

from forecasting.inference import InferenceConfig, run_openai_compatible_inference
from evaluation.evaluate import aligned_arrays, parse_forecast_value
from utils.io import atomic_write_csv


def generate_teacher_sft_csv(
    input_path: str | Path,
    output_path: str | Path,
    config: InferenceConfig,
) -> str:
    """Generate teacher responses for SFT using an OpenAI-compatible API.

    The output keeps the original prompt/ground_truth columns and writes a
    `response` column containing the full model response, matching the original
    SFT training expectation.
    """
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("teacher output must differ from its input")
    with tempfile.TemporaryDirectory(prefix="castflow_teacher_") as temporary:
        tmp_path = str(Path(temporary) / "responses.csv")
        if config.resume and Path(output_path).is_file():
            shutil.copyfile(output_path, tmp_path)
        run_openai_compatible_inference(input_path, tmp_path, config)
        frame = pd.read_csv(tmp_path, keep_default_na=False)
    if "full_response" in frame.columns:
        frame["response"] = frame["full_response"]
    elif config.output_col in frame.columns:
        frame["response"] = frame[config.output_col]
    else:
        frame["response"] = ""
    keep: list[bool] = []
    for row in frame.to_dict(orient="records"):
        forecast = parse_forecast_value(row["response"])
        valid = bool(forecast) and str(row.get("success", True)).strip().lower() in {
            "true",
            "1",
        }
        if "ground_truth" in row:
            try:
                aligned_arrays(forecast, parse_forecast_value(row["ground_truth"]))
            except (TypeError, ValueError):
                valid = False
        keep.append(valid)
    invalid_count = len(frame) - sum(keep)
    if invalid_count:
        warnings.warn(
            f"excluded {invalid_count} teacher responses with missing, malformed, or unaligned forecasts",
            stacklevel=2,
        )
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError("no valid teacher responses were produced")
    atomic_write_csv(output_path, frame)
    return str(output_path)
