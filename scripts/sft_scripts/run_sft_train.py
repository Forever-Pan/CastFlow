#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper to launch SFT training for the local LLM (Qwen3-0.6B by default).
Forwards arguments to sft_train.py and optionally enables LoRA.
"""

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run local SFT training")
    parser.add_argument("--csv-file", default="./datasets/grok_train_answers.csv", help="Training CSV with prompts and responses")
    parser.add_argument("--output-dir", default="./models/sft_model", help="Output directory for SFT weights")
    parser.add_argument("--model-name", default="./models/Qwen3-4B", help="Base model path/name") # ./models/Qwen3-0.6B，Qwen/Qwen3-4B
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size") # A800：4，4090：1
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps") # A800：1，4090：4 
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate") # 开lora：1e-4，全量微调：5e-5
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs") # 3-5轮
    parser.add_argument("--max-length", type=int, default=10000, help="Max token length")
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Checkpoint save strategy (default: epoch)")
    parser.add_argument("--save-steps", type=int, default=500, help="Steps interval when save-strategy=steps")
    parser.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints to keep")
    # Default: DeepSpeed ON, LoRA OFF. Provide flags to opt-in/out as needed.
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA (disabled by default)") # 默认关闭，提供--use_lora启用
    parser.add_argument("--no-deepspeed", action="store_true", help="Disable DeepSpeed (enabled by default)") # 默认启用，可以通过--no-deepspeed禁用
    parser.add_argument("--lora-target-modules", default="", help="Comma-separated target modules for LoRA")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "sft_train.py",
        "--csv_file",
        args.csv_file,
        "--output_dir",
        args.output_dir,
        "--model_name",
        args.model_name,
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation",
        str(args.gradient_accumulation),
        "--learning_rate",
        str(args.learning_rate),
        "--num_epochs",
        str(args.num_epochs),
        "--max_length",
        str(args.max_length),
        "--save_strategy",
        args.save_strategy,
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        str(args.save_total_limit),
    ]

    # Enable LoRA only if requested
    if args.use_lora:
        cmd.append("--use_lora")
    if args.lora_target_modules:
        cmd.extend(["--lora_target_modules", args.lora_target_modules])
    if args.use_lora:
        cmd.extend([
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
        ])

    # Enable DeepSpeed by default unless explicitly disabled
    if not args.no_deepspeed:
        cmd.append("--use_deepspeed")

    print("[SFT Train] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
