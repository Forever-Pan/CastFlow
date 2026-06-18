from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecasting.inference import InferenceConfig, run_openai_compatible_inference


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
    tmp_path = str(Path(output_path).with_suffix(".teacher_tmp.csv"))
    run_openai_compatible_inference(input_path, tmp_path, config)
    frame = pd.read_csv(tmp_path, keep_default_na=False)
    if "full_response" in frame.columns:
        frame["response"] = frame["full_response"]
    elif config.output_col in frame.columns:
        frame["response"] = frame[config.output_col]
    else:
        frame["response"] = ""
    frame = frame[frame["response"].astype(str).str.strip() != ""]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    Path(tmp_path).unlink(missing_ok=True)
    return str(output_path)
