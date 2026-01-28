#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 CSV 文件中 prompt 的字符长度和 Qwen 分词器下的 token 长度

使用方法:
    python scripts/analyze_prompt_length.py \
        --input ./results/windy/test_predictions_use_memory.csv \
        --model_path ./models/Qwen3-8B
"""

import argparse
import os
import sys
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def analyze_prompt_lengths(csv_path: str, model_path: str, prompt_column: str = 'prompt'):
    """
    分析 CSV 文件中 prompt 的字符长度和 token 长度
    
    Args:
        csv_path: CSV 文件路径
        model_path: Qwen 模型路径（用于加载分词器）
        prompt_column: prompt 列名
    """
    print(f"正在加载分词器: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ 分词器加载完成，词汇表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ 错误: 无法加载分词器: {e}")
        sys.exit(1)
    
    print(f"\n正在读取 CSV 文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
        print(f"✅ 读取完成，总行数: {len(df)}")
    except Exception as e:
        print(f"❌ 错误: 无法读取 CSV 文件: {e}")
        sys.exit(1)
    
    # 检查 prompt 列是否存在
    if prompt_column not in df.columns:
        print(f"❌ 错误: CSV 文件中没有找到 '{prompt_column}' 列")
        print(f"   可用的列: {list(df.columns)}")
        sys.exit(1)
    
    # 过滤掉空的 prompt
    df = df[df[prompt_column].str.strip() != '']
    print(f"   有效 prompt 数量: {len(df)}")
    
    # 分析每个 prompt
    print(f"\n正在分析 prompt 长度...")
    char_lengths = []
    token_lengths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        prompt = str(row[prompt_column])
        
        # 计算字符长度
        char_len = len(prompt)
        char_lengths.append(char_len)
        
        # 计算 token 长度
        try:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            token_len = len(tokens)
            token_lengths.append(token_len)
        except Exception as e:
            print(f"\n⚠️  警告: 第 {idx} 行分词失败: {e}")
            token_lengths.append(0)
    
    # 转换为 numpy 数组以便统计
    char_lengths = np.array(char_lengths)
    token_lengths = np.array(token_lengths)
    
    # 计算统计信息
    print(f"\n{'='*80}")
    print(f"Prompt 长度分析结果")
    print(f"{'='*80}")
    print(f"\n字符长度统计:")
    print(f"  总数: {len(char_lengths)}")
    print(f"  最小值: {char_lengths.min()}")
    print(f"  最大值: {char_lengths.max()}")
    print(f"  平均值: {char_lengths.mean():.2f}")
    print(f"  中位数: {np.median(char_lengths):.2f}")
    print(f"  标准差: {char_lengths.std():.2f}")
    print(f"  25%分位数: {np.percentile(char_lengths, 25):.2f}")
    print(f"  75%分位数: {np.percentile(char_lengths, 75):.2f}")
    print(f"  95%分位数: {np.percentile(char_lengths, 95):.2f}")
    print(f"  99%分位数: {np.percentile(char_lengths, 99):.2f}")
    
    print(f"\nToken 长度统计 (使用 {model_path} 分词器):")
    print(f"  总数: {len(token_lengths)}")
    print(f"  最小值: {token_lengths.min()}")
    print(f"  最大值: {token_lengths.max()}")
    print(f"  平均值: {token_lengths.mean():.2f}")
    print(f"  中位数: {np.median(token_lengths):.2f}")
    print(f"  标准差: {token_lengths.std():.2f}")
    print(f"  25%分位数: {np.percentile(token_lengths, 25):.2f}")
    print(f"  75%分位数: {np.percentile(token_lengths, 75):.2f}")
    print(f"  95%分位数: {np.percentile(token_lengths, 95):.2f}")
    print(f"  99%分位数: {np.percentile(token_lengths, 99):.2f}")
    
    # 计算字符到 token 的比率
    ratios = token_lengths / (char_lengths + 1e-10)  # 避免除零
    print(f"\n字符到 Token 比率统计:")
    print(f"  平均值: {ratios.mean():.4f}")
    print(f"  中位数: {np.median(ratios):.4f}")
    print(f"  最小值: {ratios.min():.4f}")
    print(f"  最大值: {ratios.max():.4f}")
    print(f"  (即平均每个字符约等于 {ratios.mean():.4f} 个 token)")
    
    # 显示一些示例
    print(f"\n{'='*80}")
    print(f"示例 (前 5 个 prompt):")
    print(f"{'='*80}")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        prompt = str(row[prompt_column])
        char_len = char_lengths[idx]
        token_len = token_lengths[idx]
        ratio = ratios[idx]
        
        print(f"\n示例 {idx + 1} (idx={row.get('idx', idx)}):")
        print(f"  字符长度: {char_len}")
        print(f"  Token 长度: {token_len}")
        print(f"  比率: {ratio:.4f}")
        print(f"  Prompt 预览 (前200字符):")
        print(f"    {prompt[:200]}...")
    
    # 找出最长的 prompt
    max_token_idx = token_lengths.argmax()
    max_char_idx = char_lengths.argmax()
    
    print(f"\n{'='*80}")
    print(f"最长 Prompt 示例:")
    print(f"{'='*80}")
    
    print(f"\nToken 最长的 prompt (索引: {max_token_idx}):")
    print(f"  字符长度: {char_lengths[max_token_idx]}")
    print(f"  Token 长度: {token_lengths[max_token_idx]}")
    print(f"  Prompt 预览 (前500字符):")
    print(f"    {str(df.iloc[max_token_idx][prompt_column])[:500]}...")
    
    if max_char_idx != max_token_idx:
        print(f"\n字符最长的 prompt (索引: {max_char_idx}):")
        print(f"  字符长度: {char_lengths[max_char_idx]}")
        print(f"  Token 长度: {token_lengths[max_char_idx]}")
        print(f"  Prompt 预览 (前500字符):")
        print(f"    {str(df.iloc[max_char_idx][prompt_column])[:500]}...")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='分析 CSV 文件中 prompt 的字符长度和 Qwen 分词器下的 token 长度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='./results/windy/test_predictions_use_memory.csv',
        help='输入的 CSV 文件路径'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./models/Qwen3-8B',
        help='Qwen 模型路径（用于加载分词器）'
    )
    
    parser.add_argument(
        '--prompt_column',
        type=str,
        default='prompt',
        help='CSV 文件中 prompt 列的名称'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    analyze_prompt_lengths(
        csv_path=args.input,
        model_path=args.model_path,
        prompt_column=args.prompt_column
    )


if __name__ == '__main__':
    main()

