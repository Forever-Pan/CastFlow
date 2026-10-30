from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from training.rewards import (
    baseline_prediction_dict,
    build_small_model_prediction_dict,
    canonical_dataset_name,
    parse_json_column,
    serialize_forecast,
)


@dataclass(slots=True)
class RLVRConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    validation_data_path: str | None = None
    port: int = 30549
    train_batch_size: int = 8
    rollout_n: int = 8
    n_runners: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 150
    repetition_penalty: float = 1.0
    learning_rate: float = 2e-6
    total_epochs: int = 3
    max_prompt_length: int = 10000
    max_response_length: int = 5000
    n_gpus_per_node: int = 2
    ppo_micro_batch_size_per_gpu: int = 1
    log_prob_micro_batch_size_per_gpu: int = 1
    ref_log_prob_micro_batch_size_per_gpu: int = 1
    gpu_memory_utilization: float = 0.6
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 35000
    save_freq: int = 100
    test_freq: int = 1000
    rollout_output_dir: str | None = None

    def validate(self) -> None:
        for field in (
            "port",
            "train_batch_size",
            "rollout_n",
            "n_runners",
            "total_epochs",
            "max_prompt_length",
            "max_response_length",
            "n_gpus_per_node",
            "ppo_micro_batch_size_per_gpu",
            "log_prob_micro_batch_size_per_gpu",
            "ref_log_prob_micro_batch_size_per_gpu",
            "max_num_seqs",
            "max_num_batched_tokens",
            "save_freq",
            "test_freq",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.port > 65535 or self.rollout_n < 2:
            raise ValueError("port must be <=65535 and GRPO rollout_n must be >=2")
        for field in ("temperature", "repetition_penalty", "learning_rate"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not 0 < self.top_p <= 1 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError(
                "top_p must be in (0,1] and gpu_memory_utilization in (0,1)"
            )
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k == 0
            or self.top_k < -1
        ):
            raise ValueError("top_k must be -1 or a positive integer")
        if not self.validation_data_path:
            raise ValueError(
                "RLVR requires a separate held-out --validation-path; training data are not reused for validation"
            )
        if (
            Path(self.dataset_path).resolve()
            == Path(self.validation_data_path).resolve()
        ):
            raise ValueError("training and validation paths must differ")


def prepare_rl_data(
    input_path: str | Path, output_path: str | Path, dataset_name: str = ""
) -> str:
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("RL data output must differ from its input")
    frame = normalize_rlvr_frame(
        pd.read_csv(input_path, keep_default_na=False),
        dataset_name or infer_dataset_name(input_path),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".parquet", dir=target.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(target)


def infer_dataset_name(path: str | Path) -> str:
    tokens = re.split(r"[^a-z0-9]+", Path(path).stem.lower())
    names: set[str] = set()
    for token in tokens:
        try:
            names.add(canonical_dataset_name(token))
        except ValueError:
            pass
    return next(iter(names)) if len(names) == 1 else ""


def normalize_rlvr_frame(frame: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
    missing = {"prompt", "ground_truth"}.difference(frame.columns)
    if missing:
        raise ValueError(
            f"RLVR data are missing required columns: {sorted(missing)}; ground_truth must contain true labels, never teacher responses"
        )
    if frame.empty:
        raise ValueError("RLVR dataset is empty")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(frame.to_dict(orient="records")):
        prompt = source["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"RLVR row {index} has an empty/non-text prompt")
        truth = parse_json_column(source["ground_truth"])
        if not truth:
            raise ValueError(
                f"RLVR row {index} requires a complete finite ground_truth forecast"
            )
        raw_name = source.get("dataset_name")
        name = canonical_dataset_name(
            raw_name if isinstance(raw_name, str) and raw_name.strip() else dataset_name
        )
        tool = source.get("tool", "")
        raw_prior = source.get("baseline_forecast")
        if isinstance(raw_prior, str) and not raw_prior.strip():
            raw_prior = None
        prior = (
            baseline_prediction_dict(raw_prior, truth)
            if raw_prior is not None
            else build_small_model_prediction_dict(tool, truth)
        )
        if not prior:
            raise ValueError(
                f"RLVR row {index} requires a complete target-aligned ensemble prior for the hybrid reward"
            )
        extra = normalize_extra_info(source.get("extra_info"), index)
        extra["index"] = index
        rows.append(
            {
                "idx": str(source.get("idx", index)),
                "prompt": prompt,
                "ground_truth": serialize_forecast(truth),
                "dataset_name": name,
                "tool": tool if isinstance(tool, str) else serialize_forecast(tool),
                "baseline_forecast": serialize_forecast(prior) if prior else "",
                "data_source": "castflow",
                "ability": "time_series_forecasting",
                "reward_model": "training.rewards.compute_contrastive_reward",
                "extra_info": extra,
            }
        )
    return pd.DataFrame(rows)


def build_agent_lightning_config(config: RLVRConfig) -> dict[str, Any]:
    """AgentLightning 0.3.0 / VERL 0.6.1 single-turn GRPO configuration."""
    config.validate()
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_ctrl": {"type": "fixed", "kl_coef": 0.0},
        },
        "data": {
            "train_files": config.dataset_path,
            "val_files": config.validation_data_path,
            "train_batch_size": config.train_batch_size,
            "max_prompt_length": config.max_prompt_length,
            "max_response_length": config.max_response_length,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "n": config.rollout_n,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "prompt_length": config.max_prompt_length,
                "response_length": config.max_response_length,
                "max_model_len": config.max_prompt_length + config.max_response_length,
                "log_prob_micro_batch_size_per_gpu": config.log_prob_micro_batch_size_per_gpu,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_num_seqs": config.max_num_seqs,
                "max_num_batched_tokens": config.max_num_batched_tokens,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
            },
            "actor": {
                "ppo_mini_batch_size": config.train_batch_size,
                "ppo_micro_batch_size_per_gpu": config.ppo_micro_batch_size_per_gpu,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "grad_clip": 2.0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 10.0,
                "optim": {"lr": config.learning_rate, "weight_decay": 0.01},
                "checkpoint": {
                    "save_contents": ["model", "optimizer", "extra", "hf_model"],
                    "load_contents": ["model", "optimizer", "extra"],
                    "async_save": False,
                },
                "fsdp_config": {
                    "model_dtype": "bf16",
                    "wrap_policy": {"min_num_params": 100000000},
                    "param_offload": False,
                    "optimizer_offload": False,
                    "fsdp_size": -1,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": config.ref_log_prob_micro_batch_size_per_gpu,
                "fsdp_config": {"model_dtype": "bf16", "param_offload": False},
            },
            "model": {
                "path": config.model_path,
                "trust_remote_code": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": config.n_gpus_per_node,
            "nnodes": 1,
            "val_before_train": False,
            "logger": ["console"],
            "project_name": "CastFlow",
            "experiment_name": "castflow_grpo",
            "test_freq": config.test_freq,
            "total_epochs": config.total_epochs,
            "save_freq": config.save_freq,
            "default_local_dir": config.output_dir,
        },
    }


def train_rlvr(config: RLVRConfig) -> None:
    config.validate()
    for label, path in (
        ("training dataset", config.dataset_path),
        ("validation dataset", config.validation_data_path),
        ("model", config.model_path),
    ):
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"RLVR {label} not found: {path}")
    train_frame = load_rlvr_dataframe(config.dataset_path)
    val_frame = load_rlvr_dataframe(str(config.validation_data_path))
    train_prompts = {hashlib.sha256(p.encode()).digest() for p in train_frame["prompt"]}
    if any(
        hashlib.sha256(p.encode()).digest() in train_prompts
        for p in val_frame["prompt"]
    ):
        raise ValueError(
            "RLVR validation prompts overlap training prompts; provide a held-out split"
        )
    if len(train_frame) < config.train_batch_size:
        raise ValueError("RLVR training dataset is smaller than train_batch_size")
    configure_training_environment()
    validate_rlvr_hardware(config.n_gpus_per_node)
    try:
        import agentlightning as agl
        from agentlightning.adapter import LlmProxyTraceToTriplet
    except ImportError as exc:
        raise RuntimeError(
            "RLVR requires agentlightning==0.3.0, verl==0.6.1, and their training dependencies"
        ) from exc
    for package, expected in (
        ("agentlightning", "0.3.0"),
        ("verl", "0.6.1"),
        ("vllm", "0.10.2"),
    ):
        if version(package) != expected:
            raise RuntimeError(
                f"this RLVR integration requires {package}=={expected}; install the pinned training dependencies"
            )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    for split, frame in (("training", train_frame), ("validation", val_frame)):
        for index, prompt in enumerate(frame["prompt"]):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(tokens) > config.max_prompt_length:
                raise ValueError(
                    f"RLVR {split} prompt {index} exceeds max_prompt_length={config.max_prompt_length}; shorten the evidence context or increase the explicit limit"
                )
    from training.agent import LiteAgent
    from training.lightning import ForecastingDaemon

    algorithm = agl.VERL(
        build_agent_lightning_config(config), daemon_cls=ForecastingDaemon
    )
    trainer = agl.Trainer(
        n_runners=config.n_runners,
        algorithm=algorithm,
        adapter=LlmProxyTraceToTriplet(),
        tracer=agl.OtelTracer(),
        port=config.port,
    )
    trainer.fit(
        LiteAgent(
            rollout_output_dir=config.rollout_output_dir,
            max_response_length=config.max_response_length,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        ),
        train_dataset=train_frame.to_dict(orient="records"),
        val_dataset=val_frame.to_dict(orient="records"),
    )


def configure_training_environment() -> None:
    from training.sft import normalize_training_environment

    normalize_training_environment()


def validate_rlvr_hardware(n_gpus: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RLVR requires CUDA-enabled NVIDIA GPUs; --print-config remains available without a GPU"
        )
    if torch.cuda.device_count() < n_gpus:
        raise RuntimeError(
            f"RLVR requested {n_gpus} GPUs but only {torch.cuda.device_count()} are visible; set CUDA_VISIBLE_DEVICES and --n-gpus-per-node consistently"
        )
    for index in range(n_gpus):
        if torch.cuda.get_device_capability(index)[0] < 8:
            raise RuntimeError(
                "the pinned asynchronous BF16 RLVR backend requires NVIDIA Ampere or newer GPUs (compute capability >= 8.0)"
            )


def load_rlvr_dataframe(dataset_path: str) -> pd.DataFrame:
    return normalize_rlvr_frame(
        pd.read_parquet(dataset_path), infer_dataset_name(dataset_path)
    )


def normalize_extra_info(value: object, idx: int) -> dict[str, Any]:
    if isinstance(value, dict):
        out = dict(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            out = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            out = {}
    else:
        out = {}
    out.setdefault("index", idx)
    return out


def parse_rlvr_config(args: argparse.Namespace) -> RLVRConfig:
    return RLVRConfig(
        dataset_path=args.dataset_path,
        validation_data_path=getattr(args, "validation_path", None),
        model_path=args.model_path,
        output_dir=args.output_dir,
        port=args.port,
        train_batch_size=args.train_batch_size,
        rollout_n=args.rollout_n,
        n_runners=args.n_runners,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        learning_rate=args.learning_rate,
        total_epochs=args.total_epochs,
        n_gpus_per_node=args.n_gpus_per_node,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        rollout_output_dir=args.rollout_output_dir,
        max_prompt_length=getattr(args, "max_prompt_length", 10000),
        max_response_length=getattr(args, "max_response_length", 5000),
    )
