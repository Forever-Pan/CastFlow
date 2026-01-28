#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate teacher answers on the test/validation split using validate_sft.
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from validate_sft import validate_sft


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher outputs")
    parser.add_argument(
        "--csv-file",
        default="./datasets/grok_test_answers.csv",
        help="CSV produced by teacher_generate_test.py",
    )
    args = parser.parse_args()

    validate_sft(args.csv_file)


if __name__ == "__main__":
    main()
