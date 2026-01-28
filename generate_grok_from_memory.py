#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grok.csv from memory file and training data.
Combines memory data with ground truth from training set.

Usage:
    python generate_grok_from_memory.py \
        --memory_file memory/windy/windy_memory.json \
        --train_file datasets/windy_power_train.csv \
        --output_file datasets/past/windy/grok.csv \
        --date_column date \
        --target_column OT \
        --forecast_length 96 \
        --time_interval 15
"""

import json
import pandas as pd
import re
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import csv
import os

def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

def get_next_timestamps(last_timestamp: str, forecast_length: int = 96, time_interval: int = 15) -> List[str]:
    """Generate next N timestamps (time_interval minutes apart)."""
    last_dt = parse_timestamp(last_timestamp)
    timestamps = []
    for i in range(forecast_length):
        next_dt = last_dt + timedelta(minutes=time_interval * (i + 1))
        timestamps.append(next_dt.strftime("%Y-%m-%d %H:%M:%S"))
    return timestamps

def extract_ground_truth(
    input_json_str: str, 
    train_df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'OT',
    forecast_length: int = 96,
    time_interval: int = 15,
    skip_missing: bool = False,
    verbose: bool = False
) -> Optional[str]:
    """Extract ground truth from training data based on input timestamps."""
    # Parse input JSON to get the last timestamp
    try:
        input_dict = json.loads(input_json_str)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Error parsing input JSON: {e}")
        return None
    
    timestamps = sorted(input_dict.keys())
    if not timestamps:
        if verbose:
            print("Warning: No timestamps found in input JSON")
        return None
    
    last_timestamp = timestamps[-1]
    
    # Generate next timestamps
    next_timestamps = get_next_timestamps(last_timestamp, forecast_length, time_interval)
    
    # Create a mapping from date string to target value
    train_df['date_str'] = train_df[date_column].astype(str)
    date_to_value = dict(zip(train_df['date_str'], train_df[target_column]))
    
    # Build ground truth dictionary
    ground_truth_dict = {}
    missing_count = 0
    for ts in next_timestamps:
        if ts in date_to_value:
            ground_truth_dict[ts] = float(date_to_value[ts])
        else:
            missing_count += 1
            if skip_missing:
                if verbose:
                    print(f"Warning: Timestamp {ts} not found in training data, skipping")
                continue
            else:
                if verbose:
                    print(f"Warning: Timestamp {ts} not found in training data, using 0.0")
                ground_truth_dict[ts] = 0.0
    
    if missing_count > 0 and verbose:
        print(f"Warning: {missing_count} out of {forecast_length} timestamps not found in training data")
    
    if not ground_truth_dict:
        if verbose:
            print("Error: No ground truth data extracted")
        return None
    
    # Convert to JSON string with double quotes escaped for CSV
    return json.dumps(ground_truth_dict, ensure_ascii=False)

def extract_reasoning_and_answer(forecasting_answer: str) -> tuple:
    """Extract reasoning and answer from Forecasting_answer."""
    if not forecasting_answer:
        return "", ""
    
    # Extract content between <think> and </think>
    think_match = re.search(r'<think>(.*?)</think>', forecasting_answer, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # Extract content between <answer> and </answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', forecasting_answer, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    return reasoning, answer

def format_tool_calls(tool_calls: List[Dict]) -> str:
    """Format tool_calls to JSON string."""
    if not tool_calls:
        return ""
    return json.dumps(tool_calls, ensure_ascii=False)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate grok.csv from memory file and training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python generate_grok_from_memory.py \\
      --memory_file memory/windy/windy_memory.json \\
      --train_file datasets/windy_power_train.csv \\
      --output_file datasets/past/windy/grok.csv

  # Custom column names
  python generate_grok_from_memory.py \\
      --memory_file memory/windy/windy_memory.json \\
      --train_file datasets/windy_power_train.csv \\
      --output_file datasets/past/windy/grok.csv \\
      --date_column timestamp \\
      --target_column power

  # Different forecast length and time interval
  python generate_grok_from_memory.py \\
      --memory_file memory/windy/windy_memory.json \\
      --train_file datasets/windy_power_train.csv \\
      --output_file datasets/past/windy/grok.csv \\
      --forecast_length 48 \\
      --time_interval 30

  # Verbose mode with skip missing timestamps
  python generate_grok_from_memory.py \\
      --memory_file memory/windy/windy_memory.json \\
      --train_file datasets/windy_power_train.csv \\
      --output_file datasets/past/windy/grok.csv \\
      --verbose \\
      --skip_missing
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--memory_file',
        type=str,
        required=True,
        help='Path to memory JSON file'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='Path to training CSV file containing ground truth data'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--date_column',
        type=str,
        default='date',
        help='Name of the date/timestamp column in training file (default: date)'
    )
    parser.add_argument(
        '--target_column',
        type=str,
        default='OT',
        help='Name of the target column in training file (default: OT for ETT/EPF-NP/EPF-BE/EPF-DE/EPF-FR/EPF-PJM; for windy/sunny use real_power)'
    )
    parser.add_argument(
        '--forecast_length',
        type=int,
        default=96,
        help='Number of time points to forecast (default: 96; for EPF-NP/EPF-BE/EPF-DE/EPF-FR/EPF-PJM use 24)'
    )
    parser.add_argument(
        '--time_interval',
        type=int,
        default=15,
        help='Time interval in minutes between consecutive points (default: 15 for ETTm1; for MOPEX use 1440/daily; for ETTh1 use 60; for EPF-NP/EPF-BE/EPF-DE/EPF-FR/EPF-PJM use 60; windy/sunny as needed)'
    )
    parser.add_argument(
        '--skip_missing',
        action='store_true',
        help='Skip missing timestamps instead of using 0.0 (default: False)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (default: False)'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate input files exist
    if not os.path.exists(args.memory_file):
        print(f"Error: Memory file not found: {args.memory_file}")
        return
    
    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found: {args.train_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if args.verbose:
            print(f"Created output directory: {output_dir}")
    
    print(f"Loading memory file: {args.memory_file}")
    try:
        with open(args.memory_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
    except Exception as e:
        print(f"Error loading memory file: {e}")
        return
    
    print(f"Loading training data: {args.train_file}")
    try:
        train_df = pd.read_csv(args.train_file)
        
        # Validate columns exist
        if args.date_column not in train_df.columns:
            print(f"Error: Date column '{args.date_column}' not found in training file")
            print(f"Available columns: {list(train_df.columns)}")
            return
        
        if args.target_column not in train_df.columns:
            print(f"Error: Target column '{args.target_column}' not found in training file")
            print(f"Available columns: {list(train_df.columns)}")
            return
        
        train_df[args.date_column] = pd.to_datetime(train_df[args.date_column])
    except Exception as e:
        print(f"Error loading training file: {e}")
        return
    
    print(f"Found {len(memory_data)} records in memory file")
    print(f"Found {len(train_df)} records in training file")
    if args.verbose:
        print(f"Using date column: {args.date_column}")
        print(f"Using target column: {args.target_column}")
        print(f"Forecast length: {args.forecast_length}")
        print(f"Time interval: {args.time_interval} minutes")
    
    # Process each memory record
    output_rows = []
    skipped_count = 0
    for i, record in enumerate(memory_data):
        idx = record.get('idx')
        input_json = record.get('input_json', '')
        tool_calls = record.get('tool_calls', [])
        forecast_full_prompt = record.get('forecast_full_prompt', '')
        forecasting_answer = record.get('Forecasting_answer', '')
        
        if args.verbose and (i + 1) % 50 == 0:
            print(f"Processing record {i + 1}/{len(memory_data)}...")
        
        # Extract ground truth from training data
        try:
            ground_truth = extract_ground_truth(
                input_json, 
                train_df,
                date_column=args.date_column,
                target_column=args.target_column,
                forecast_length=args.forecast_length,
                time_interval=args.time_interval,
                skip_missing=args.skip_missing,
                verbose=args.verbose
            )
            
            if ground_truth is None:
                skipped_count += 1
                if args.verbose:
                    print(f"Skipping record idx {idx}: failed to extract ground truth")
                continue
        except Exception as e:
            skipped_count += 1
            print(f"Error extracting ground truth for idx {idx}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
        
        # Format tool calls
        tool = format_tool_calls(tool_calls)
        
        # Extract reasoning and answer
        reasoning, answer = extract_reasoning_and_answer(forecasting_answer)
        
        # Build row
        row = {
            'idx': idx,
            'input': input_json,
            'ground_truth': ground_truth,
            'tool': tool,
            'prompt': forecast_full_prompt,
            'response': forecasting_answer,
            'reasoning': reasoning,
            'answer': answer
        }
        output_rows.append(row)
    
    # Sort by idx
    output_rows.sort(key=lambda x: x['idx'])
    
    print(f"\nGenerated {len(output_rows)} rows")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} records due to errors")
    
    # Write to CSV
    print(f"Writing to {args.output_file}...")
    fieldnames = ['idx', 'input', 'ground_truth', 'tool', 'prompt', 'response', 'reasoning', 'answer']
    
    try:
        with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(output_rows)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return
    
    if output_rows:
        print(f"\nDone! Generated {len(output_rows)} records in {args.output_file}")
        print(f"Index range: {min(r['idx'] for r in output_rows)} to {max(r['idx'] for r in output_rows)}")
    else:
        print("\nWarning: No records were generated!")

if __name__ == "__main__":
    main()
