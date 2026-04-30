from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RESPONSE_ALIASES = ("response", "full_answer", "answer_teacher", "full_response", "answer_local_llm", "answer")


@dataclass(slots=True)
class SFTConfig:
    dataset_path: str
    model_path: str
    output_dir: str
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 1
    max_steps: int = -1
    max_length: int = 14000
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 5
    use_deepspeed: bool = True
    stage3_gather_on_save: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    cuda_visible_devices: str | None = None


def load_prompt_response_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df = normalize_sft_frame(df)
    df = df[df["prompt"].astype(str).str.strip() != ""]
    df = df[df["response"].astype(str).str.strip() != ""]
    if df.empty:
        raise ValueError("SFT dataset has no valid prompt/response rows")
    return df.reset_index(drop=True)


def normalize_sft_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "prompt" not in df.columns:
        raise ValueError(f"SFT dataset missing prompt column; found {list(df.columns)}")
    response_col = next((col for col in RESPONSE_ALIASES if col in df.columns), None)
    if response_col is None:
        raise ValueError(
            f"SFT dataset missing response column; accepted aliases={list(RESPONSE_ALIASES)}; "
            f"found {list(df.columns)}"
        )
    out = df.copy()
    out["response"] = out[response_col].astype(str)
    return out


def prepare_sft_csv(input_path: str | Path, output_path: str | Path) -> str:
    """Normalize old/evaluation CSVs into prompt,response SFT format."""
    df = normalize_sft_frame(pd.read_csv(input_path, keep_default_na=False))
    out = df[["prompt", "response"]].copy()
    out = out[out["prompt"].astype(str).str.strip() != ""]
    out = out[out["response"].astype(str).str.strip() != ""]
    if out.empty:
        raise ValueError("prepared SFT dataset has no valid prompt/response rows")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return str(output_path)


def row_to_messages(row: pd.Series) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": str(row["prompt"]).strip()},
            {"role": "assistant", "content": str(row["response"]).strip()},
        ]
    }


def train_sft(config: SFTConfig) -> None:
    """Run supervised fine-tuning with optional LoRA.

    Heavy dependencies are imported lazily so normal CastFlow forecasting remains lightweight.
    """
    normalize_training_environment(config.cuda_visible_devices)
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"SFT dataset not found: {config.dataset_path}")
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"model path not found: {config.model_path}")

    try:
        import torch
        from datasets import Dataset, disable_progress_bars
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "SFT training requires torch, datasets, and transformers. "
            "Install CastFlow training dependencies before running train-sft."
        ) from exc

    if int(os.environ.get("RANK", "0")) != 0:
        disable_progress_bars()

    peft_available = False
    ds_config = build_deepspeed_config(config) if config.use_deepspeed else None

    if config.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_available = True
        except ImportError as exc:
            raise RuntimeError("LoRA SFT requires peft") from exc

    df = load_prompt_response_csv(config.dataset_path)
    dataset = Dataset.from_list([row_to_messages(row) for _, row in df.iterrows()])

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_chat(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=config.max_length, padding=False, add_special_tokens=False)
        labels = list(encoded["input_ids"])
        prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        for idx in range(min(prompt_len, len(labels))):
            labels[idx] = -100
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}

    tokenized = dataset.map(tokenize_chat, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if config.use_lora and peft_available:
        targets = config.lora_target_modules or infer_lora_targets(config.model_path)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=targets,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    class DataCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            labels = [torch.tensor(f["labels"]) for f in features]
            attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
                "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            }

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=[],
        deepspeed=ds_config,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=DataCollator())
    trainer.train()
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.output_dir)


def normalize_training_environment(cuda_visible_devices: str | None = None) -> None:
    """Clean up CUDA/DeepSpeed env vars inherited from shells or notebooks."""
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cuda_home = os.environ.get("CUDA_HOME", "")
    candidates: list[Path] = []
    for part in cuda_home.split(os.pathsep):
        cleaned = part.strip()
        if cleaned:
            candidates.append(Path(cleaned))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.append(Path("/usr/local/cuda"))

    for candidate in candidates:
        if (candidate / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(candidate)
            os.environ.setdefault("CUDA_PATH", str(candidate))
            break

    triton_cache = Path(os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/castflow_triton_cache"))
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DISABLED", "true")


def build_deepspeed_config(config: SFTConfig) -> dict[str, Any] | None:
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        return None

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    train_batch_size = max(1, world_size) * config.batch_size * config.gradient_accumulation
    return {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": config.stage3_gather_on_save,
        },
        "gradient_accumulation_steps": config.gradient_accumulation,
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": config.batch_size,
        "wall_clock_breakdown": False,
    }


def infer_lora_targets(model_path: str) -> list[str]:
    name = model_path.lower()
    if "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_sft_config(args: argparse.Namespace) -> SFTConfig:
    targets = args.lora_target_modules.split(",") if args.lora_target_modules else None
    return SFTConfig(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_deepspeed=args.use_deepspeed,
        stage3_gather_on_save=args.stage3_gather_on_save,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=targets,
        cuda_visible_devices=args.cuda_visible_devices,
    )
