from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class RLVRConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    port: int = 30549
    train_batch_size: int = 8
    rollout_n: int = 8
    n_runners: int = 64
    temperature: float = 1
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


def prepare_rl_data(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str = "",
) -> str:
    df = pd.read_csv(input_path, keep_default_na=False)
    if "prompt" not in df.columns:
        raise ValueError(f"RL data requires a prompt column; found {list(df.columns)}")
    if "ground_truth" not in df.columns:
        if "response" in df.columns:
            df["ground_truth"] = df["response"]
        else:
            df["ground_truth"] = ""
    elif "response" in df.columns:
        empty_ground_truth = df["ground_truth"].astype(str).str.strip() == ""
        df.loc[empty_ground_truth, "ground_truth"] = df.loc[empty_ground_truth, "response"]
    if "tool" not in df.columns:
        df["tool"] = ""
    if "idx" not in df.columns:
        df["idx"] = df.index

    resolved_dataset = dataset_name or infer_dataset_name(input_path)
    if "dataset_name" not in df.columns:
        df["dataset_name"] = resolved_dataset
    else:
        df["dataset_name"] = df["dataset_name"].replace("", resolved_dataset)

    keep = [
        col
        for col in [
            "idx",
            "prompt",
            "response",
            "ground_truth",
            "answer",
            "tool",
            "dataset_name",
            "baseline_forecast",
        ]
        if col in df.columns
    ]
    out = df[keep].copy()
    out = out[out["prompt"].astype(str).str.strip() != ""]
    out = out[out["ground_truth"].astype(str).str.strip() != ""]
    out["data_source"] = "castflow"
    out["ability"] = "time_series_forecasting"
    out["reward_model"] = "scripts.training.rewards.compute_contrastive_reward"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return str(output_path)


def infer_dataset_name(path: str | Path) -> str:
    text = str(path).lower()
    aliases = {
        "windy": "windy",
        "wind": "windy",
        "sunny": "sunny",
        "solar": "sunny",
        "mopex": "MOPEX",
        "etth1": "ETTh1",
        "ettm1": "ETTm1",
        "pjm": "PJM",
        "np": "NP",
        "be": "BE",
        "de": "DE",
        "fr": "FR",
    }
    for key, value in aliases.items():
        if key in text:
            return value
    return ""


def build_agent_lightning_config(config: RLVRConfig) -> dict[str, Any]:
    """Build the AgentLightning/VERL config migrated from rl_agent_train.py."""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": True,
            "kl_coeff": 0.0,
            "kl_ctrl": {"type": "fixed", "kl_coef": 0.0},
        },
        "data": {
            "train_files": config.dataset_path,
            "val_files": config.dataset_path,
            "train_batch_size": config.train_batch_size,
            "max_prompt_length": config.max_prompt_length,
            "max_response_length": config.max_response_length,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "rollout": {
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "n": config.rollout_n,
                "log_prob_micro_batch_size_per_gpu": config.log_prob_micro_batch_size_per_gpu,
                "name": "vllm",
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_num_seqs": config.max_num_seqs,
                "max_num_batched_tokens": config.max_num_batched_tokens,
                "enable_chunked_prefill": True,
                "sampling_params": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repetition_penalty": config.repetition_penalty,
                },
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
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
                    "save_contents": ["hf_model"],
                    "load_contents": ["hf_model"],
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
                "fsdp_config": {
                    "model_dtype": "bf16",
                    "wrap_policy": {"min_num_params": 100000000},
                    "param_offload": False,
                },
            },
            "model": {"path": config.model_path, "trust_remote_code": True, "enable_gradient_checkpointing": True},
        },
        "trainer": {
            "n_gpus_per_node": config.n_gpus_per_node,
            "val_before_train": False,
            "logger": ["console", "wandb"],
            "project_name": "VLMTimeSSeriesAgent",
            "experiment_name": "vlm_tsss_training",
            "nnodes": 1,
            "test_freq": config.test_freq,
            "total_epochs": config.total_epochs,
            "save_freq": config.save_freq,
            "default_local_dir": config.output_dir,
        },
    }


def train_rlvr(config: RLVRConfig) -> None:
    """Run RLVR/GRPO training via the migrated rl_agent.py + rl_agent_train.py flow."""
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"RLVR dataset not found: {config.dataset_path}")
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"model path not found: {config.model_path}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    configure_training_environment()

    try:
        import agentlightning as agl
        from agentlightning.adapter import TracerTraceToTriplet
    except ImportError as exc:
        raise RuntimeError(
            "RLVR training requires agentlightning and its TracerTraceToTriplet adapter."
        ) from exc

    from .agent import LiteAgent

    train_df = load_rlvr_dataframe(config.dataset_path)
    val_df = load_rlvr_dataframe(config.dataset_path)
    train_data = train_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")

    algorithm = agl.VERL(build_agent_lightning_config(config))
    adapter = TracerTraceToTriplet(
        agent_match=None,
        llm_call_match=r"openai\.chat\.completion",
        _skip_empty_token_spans=True,
    )
    trainer = agl.Trainer(
        n_runners=config.n_runners,
        algorithm=algorithm,
        adapter=adapter,
        port=config.port,
    )
    trainer.fit(LiteAgent(rollout_output_dir=config.rollout_output_dir), train_dataset=train_data, val_dataset=val_data)


def configure_training_environment() -> None:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        cuda_home = str(Path(nvcc_path).parent.parent)
        os.environ["CUDA_HOME"] = cuda_home
    else:
        for cuda_path in ["/usr/local/cuda", "/usr/local/cuda-12.6", "/usr/local/cuda-12", "/usr/local/cuda-11.8"]:
            nvcc_file = Path(cuda_path) / "bin" / "nvcc"
            if nvcc_file.exists():
                os.environ["CUDA_HOME"] = cuda_path
                os.environ["PATH"] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
                break

    if "CUDA_HOME" in os.environ:
        cuda_home = os.environ["CUDA_HOME"].strip()
        if cuda_home.startswith(":"):
            cuda_home = cuda_home[1:]
        if ":" in cuda_home:
            cuda_home = cuda_home.split(":")[0]
        os.environ["CUDA_HOME"] = cuda_home

    if "TRITON_CACHE_DIR" not in os.environ:
        os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_" + getpass.getuser()
    os.environ["WANDB_MODE"] = "offline"


def load_rlvr_dataframe(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    if len(df) == 0:
        raise ValueError(f"RLVR dataset is empty: {dataset_path}")
    required_columns = ["prompt", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"RLVR dataset is missing required columns: {missing_columns}")
    if "dataset_name" not in df.columns:
        df = df.copy()
        df["dataset_name"] = infer_dataset_name(dataset_path)
    if "tool" not in df.columns:
        df = df.copy()
        df["tool"] = ""
    if "extra_info" in df.columns:
        df = df.copy()
        df["extra_info"] = [normalize_extra_info(value, idx) for idx, value in enumerate(df["extra_info"])]
    return df


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
    if "index" not in out:
        out["index"] = out.get("idx", idx)
    return out


def parse_rlvr_config(args: argparse.Namespace) -> RLVRConfig:
    return RLVRConfig(
        dataset_path=args.dataset_path,
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
    )
