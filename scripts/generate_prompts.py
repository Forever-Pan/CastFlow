#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompt CSVs for SFT and RL.
Build sliding-window samples for train and test splits using process_data.process_data.
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# 硬编码：可直接修改以下常量，无需每次传参
# 例如 EPF-NP：LOOK_BACK=168, GROUND_TRUTH_LENGTH=24
# ---------------------------------------------------------------------------
LOOK_BACK = 96
GROUND_TRUTH_LENGTH = 96
STRIDE = None  # 滑窗步长，None 时使用 LOOK_BACK


# Ensure project root is on sys.path so process_data can be imported when run from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_data import process_data as build_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate prompt CSVs for SFT/RL.")
    parser.add_argument(
        "--train-input",
        default="./datasets/windy_power_train.csv",
        help="Source CSV for train split",
    )
    parser.add_argument(
        "--train-output",
        default="./datasets/windy_power_processed_train.csv",
        help="Output CSV with prompts for train split",
    )
    parser.add_argument(
        "--test-input",
        default="./datasets/windy_power_test.csv",
        help="Source CSV for test/validation split",
    )
    parser.add_argument(
        "--test-output",
        default="./datasets/windy_power_processed_test.csv",
        help="Output CSV with prompts for test/validation split",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip generating the test/validation split",
    )
    parser.add_argument(
        "--look-back",
        type=int,
        default=LOOK_BACK,
        help=f"历史窗口长度 / input 点数（默认: {LOOK_BACK}，EPF-NP 用 168）",
    )
    parser.add_argument(
        "--ground-truth-length",
        type=int,
        default=GROUND_TRUTH_LENGTH,
        help=f"ground_truth 长度 / 预测点数（默认: {GROUND_TRUTH_LENGTH}，EPF-NP 用 24）",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="滑窗步长；省略或为 None 时由 process_data 使用 look_back",
    )
    args = parser.parse_args()

    stride_display = args.stride if args.stride is not None else args.look_back
    print(f"[Prompts] look_back={args.look_back}, ground_truth_length={args.ground_truth_length}, stride={stride_display}")

    print("[Prompts] Generating train split...")
    build_prompts(
        args.train_input,
        args.train_output,
        look_back=args.look_back,
        ground_truth_length=args.ground_truth_length,
        stride=args.stride,
    )

    if not args.skip_test:
        print("[Prompts] Generating test/validation split...")
        build_prompts(
            args.test_input,
            args.test_output,
            look_back=args.look_back,
            ground_truth_length=args.ground_truth_length,
            stride=args.stride,
        )
    else:
        print("[Prompts] Skipped test/validation generation by request.")


if __name__ == "__main__":
    main()