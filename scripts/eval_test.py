#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate post-SFT model outputs on the test split using validate_sft.
"""

import argparse
import os
import sys
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validate_sft import validate_sft


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model outputs")
    parser.add_argument(
        "--csv-file",
        default="./results/MOPEX/SFT/0.6B_cross.csv",
        help="CSV produced by scripts/infer_test.py (contains prompt + answer_local_LLM + ground_truth)",
    )
    parser.add_argument(
        "--answer-col",
        default="answer_local_LLM",
        help="Which column to evaluate as prediction answer",
    )
    parser.add_argument(
        "--ground-truth-col",
        default="ground_truth",
        help="Which column to use as ground truth",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"❌ Error: file not found: {args.csv_file}")
        return

    df = pd.read_csv(args.csv_file, keep_default_na=False)
    if args.answer_col not in df.columns:
        print(f"❌ Error: answer column '{args.answer_col}' not found. Available columns: {list(df.columns)}")
        return
    if args.ground_truth_col not in df.columns:
        print(f"❌ Error: ground truth column '{args.ground_truth_col}' not found. Available columns: {list(df.columns)}")
        return

    # validate_sft 固定读取列名 'answer' 和 'ground_truth'
    df_prepared = df.copy()
    df_prepared["answer"] = df_prepared[args.answer_col]
    df_prepared["ground_truth"] = df_prepared[args.ground_truth_col]

    base, ext = os.path.splitext(args.csv_file)
    prepared_csv = f"{base}__prepared_for_validate_sft{ext or '.csv'}"
    df_prepared.to_csv(prepared_csv, index=False, encoding="utf-8")
    print(f"Prepared evaluation file saved to: {prepared_csv}")

    validate_sft(prepared_csv)


if __name__ == "__main__":
    main()
