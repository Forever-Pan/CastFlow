from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _complete_huggingface_model(path: Path) -> bool:
    if (
        not (path / "config.json").is_file()
        or not (path / "tokenizer_config.json").is_file()
    ):
        return False
    if not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
    ):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file() and (path / name).stat().st_size > 0:
            return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if not index.is_file():
            continue
        try:
            weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weights, dict) or not weights:
                continue
            shards = [path / shard for shard in set(weights.values())]
            if all(
                shard.resolve().is_relative_to(path.resolve())
                and shard.is_file()
                and shard.stat().st_size > 0
                for shard in shards
            ):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def resolve_forecasting_checkpoint(model_path: str | Path) -> Path:
    """Resolve a full HF model or the latest completed VERL actor export."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"forecasting model directory not found: {root}")
    for direct in (root, root / "huggingface", root / "actor" / "huggingface"):
        if _complete_huggingface_model(direct):
            return direct
    marker = root / "latest_checkpointed_iteration.txt"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if not value.isdigit():
            raise ValueError(f"invalid latest-checkpoint marker: {marker}")
        latest = root / f"global_step_{int(value)}"
    else:
        steps = [
            (int(match.group(1)), child)
            for child in root.iterdir()
            if child.is_dir()
            and (match := re.fullmatch(r"global_step_(\d+)", child.name))
        ]
        if not steps:
            raise ValueError(
                f"no complete full-model export found under {root}; specify a merged SFT or RL actor/huggingface directory"
            )
        latest = max(steps, key=lambda pair: pair[0])[1]
    candidate = latest / "actor" / "huggingface"
    if not _complete_huggingface_model(candidate):
        raise ValueError(
            f"latest RL checkpoint lacks a complete Hugging Face model/tokenizer export: {candidate}"
        )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a trained CastFlow forecasting model"
    )
    parser.add_argument("model_path")
    args = parser.parse_args()
    try:
        print(resolve_forecasting_checkpoint(args.model_path))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
