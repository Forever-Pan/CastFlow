#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate teacher (grok-4-1-fast-non-reasoning) answers on the training split.
Relies on call_api_parallel.process_csv_parallel for parallel OpenAI-compatible requests.
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from call_api_parallel import process_csv_parallel


def main():
    parser = argparse.ArgumentParser(description="Teacher inference on training data")
    parser.add_argument(
        "--input",
        default="./datasets/windy_power_processed_train.csv",
        help="Input prompt CSV (train)",
    )
    parser.add_argument(
        "--output",
        default="./datasets/grok_train_answers.csv",
        help="Output CSV with teacher responses",
    )
    parser.add_argument("--workers", type=int, default=20, help="Max parallel workers")
    parser.add_argument("--timeout", type=float, default=240.0, help="Request timeout seconds")
    parser.add_argument(
        "--model",
        default="grok-4-1-fast-non-reasoning",
        help="Teacher model name passed via MODEL env",
    )
    args = parser.parse_args()

    os.environ.setdefault("MODEL", args.model)
    process_csv_parallel(args.input, args.output, max_workers=args.workers, timeout=args.timeout)


if __name__ == "__main__":
    main()
