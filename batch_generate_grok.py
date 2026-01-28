#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成grok.csv文件的示例脚本
可以根据配置文件批量处理多个数据集
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_dataset(dataset_config, base_dir=None):
    """处理单个数据集"""
    if base_dir:
        # 将相对路径转换为绝对路径
        for key in ['memory_file', 'train_file', 'output_file']:
            if key in dataset_config and not os.path.isabs(dataset_config[key]):
                dataset_config[key] = os.path.join(base_dir, dataset_config[key])
    
    name = dataset_config.get('name', 'unknown')
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'generate_grok_from_memory.py',
        '--memory_file', dataset_config['memory_file'],
        '--train_file', dataset_config['train_file'],
        '--output_file', dataset_config['output_file'],
    ]
    
    # 添加可选参数
    if 'date_column' in dataset_config:
        cmd.extend(['--date_column', dataset_config['date_column']])
    if 'target_column' in dataset_config:
        cmd.extend(['--target_column', dataset_config['target_column']])
    if 'forecast_length' in dataset_config:
        cmd.extend(['--forecast_length', str(dataset_config['forecast_length'])])
    if 'time_interval' in dataset_config:
        cmd.extend(['--time_interval', str(dataset_config['time_interval'])])
    if dataset_config.get('skip_missing', False):
        cmd.append('--skip_missing')
    if dataset_config.get('verbose', True):
        cmd.append('--verbose')
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 输出结果
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"✗ Failed to process {name}", file=sys.stderr)
        return False
    else:
        print(f"✓ Completed: {name}")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成grok.csv文件')
    parser.add_argument(
        '--config',
        type=str,
        default='batch_config.json',
        help='配置文件路径（默认: batch_config.json）'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='基础目录路径（用于解析相对路径）'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='只处理指定的数据集（按name字段匹配）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("\nExample config file structure:")
        print("""
{
    "datasets": [
        {
            "name": "windy",
            "memory_file": "memory/windy/windy_memory.json",
            "train_file": "datasets/windy_power_train.csv",
            "output_file": "datasets/past/windy/grok.csv",
            "date_column": "date",
            "target_column": "real_power",
            "forecast_length": 96,
            "time_interval": 15,
            "verbose": true
        }
    ]
}
        """)
        return 1
    
    config = load_config(args.config)
    
    # 确定基础目录
    if args.base_dir:
        base_dir = args.base_dir
    else:
        # 使用配置文件所在目录
        base_dir = os.path.dirname(os.path.abspath(args.config))
    
    # 处理数据集
    datasets = config.get('datasets', [])
    if not datasets:
        print("Error: No datasets found in config file")
        return 1
    
    # 如果指定了数据集，只处理该数据集
    if args.dataset:
        datasets = [d for d in datasets if d.get('name') == args.dataset]
        if not datasets:
            print(f"Error: Dataset '{args.dataset}' not found in config")
            return 1
    
    print(f"Found {len(datasets)} dataset(s) to process")
    
    success_count = 0
    for dataset in datasets:
        if process_dataset(dataset, base_dir):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(datasets)} datasets processed successfully")
    print(f"{'='*60}")
    
    return 0 if success_count == len(datasets) else 1

if __name__ == "__main__":
    sys.exit(main())
