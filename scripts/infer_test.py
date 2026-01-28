#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate answers on the test split using the SFT local LLM served via vLLM.
Simple parallel processing without checkpoint/resume or temp files.
"""

import argparse
import os
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from call_api_parallel import parse_response
from validate_sft import parse_json_column, calculate_metrics
import numpy as np


def is_valid_answer(answer: str, ground_truth: str) -> bool:
    """
    检查答案是否有效（基于 validate_sft 的逻辑）
    
    Args:
        answer: 预测答案（JSON字符串）
        ground_truth: 真实值（JSON字符串）
        
    Returns:
        True 如果答案有效（可以计算MSE），False 如果无效
    """
    if not answer or not answer.strip():
        return False
    
    if not ground_truth or not ground_truth.strip():
        return False
    
    # 解析JSON
    answer_dict = parse_json_column(answer)
    gt_dict = parse_json_column(ground_truth)
    
    # 计算指标
    mse, _, _, _, _, _, _ = calculate_metrics(answer_dict, gt_dict)
    
    # 如果MSE是nan，则无效
    return not np.isnan(mse)


def call_api(prompt, idx, config, timeout, max_retries=3, ground_truth=None):
    """
    调用单个API请求，带重试机制
    
    Args:
        prompt: 输入提示
        idx: 行索引
        config: API配置
        timeout: 超时时间
        max_retries: 最大重试次数
        ground_truth: 真实值（用于验证答案有效性，可选）
    """
    import time
    import re
    
    last_error = None
    dynamic_max_tokens = config.get("max_tokens")  # 动态调整的 max_tokens
    
    for attempt in range(max_retries):
        start = time.time()
        try:
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=timeout * 2  # 增加客户端超时时间
            )
            
            params = {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "timeout": timeout * 2,  # 增加API调用超时时间
            }
            if dynamic_max_tokens:
                params["max_tokens"] = dynamic_max_tokens
            
            response = client.chat.completions.create(**params)
            response_text = (response.choices[0].message.content or "").strip()
            _, answer = parse_response(response_text)
            
            # 如果提供了ground_truth，验证答案有效性
            if ground_truth:
                if is_valid_answer(answer, ground_truth):
                    elapsed = time.time() - start
                    return {"idx": idx, "answer": answer, "success": True, "error": None, "elapsed": elapsed, "attempt": attempt + 1}
                else:
                    # 答案无效，继续重试
                    last_error = "Invalid answer (cannot calculate MSE)"
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # 最后一次尝试也无效，返回无效答案
                        elapsed = time.time() - start
                        return {"idx": idx, "answer": answer, "success": False, "error": last_error, "elapsed": elapsed, "attempt": attempt + 1}
            else:
                # 没有ground_truth，直接返回
                elapsed = time.time() - start
                return {"idx": idx, "answer": answer, "success": True, "error": None, "elapsed": elapsed, "attempt": attempt + 1}
                
        except Exception as e:
            elapsed = time.time() - start
            error_type = type(e).__name__
            error_msg = str(e)
            last_error = f"{error_type}: {error_msg}"
            
            # 如果是 max_tokens 过大的错误，尝试动态调整
            if "max_tokens" in error_msg.lower() or "max_completion_tokens" in error_msg.lower():
                # 解析错误信息，提取输入 tokens 和最大上下文长度
                # 错误格式: "max_tokens is too large: 5500. This model's maximum context length is 15000 tokens and your request has 10931 input tokens (5500 > 15000 - 10931)."
                match = re.search(r'maximum context length is (\d+) tokens and your request has (\d+) input tokens', error_msg)
                if match:
                    max_context_len = int(match.group(1))
                    input_tokens = int(match.group(2))
                    # 计算可用的 max_tokens，留一些余量（减去 100 tokens 作为安全余量）
                    available_tokens = max_context_len - input_tokens - 100
                    if available_tokens > 0:
                        dynamic_max_tokens = min(available_tokens, dynamic_max_tokens or 5500)
                        if attempt < max_retries - 1:
                            # 使用调整后的 max_tokens 重试
                            continue
                    else:
                        # 输入 tokens 已经超过或接近最大上下文长度
                        last_error = f"Input tokens ({input_tokens}) exceed or too close to max context length ({max_context_len})"
            
            if attempt < max_retries - 1:
                # 继续重试
                continue
            else:
                # 最后一次尝试也失败
                return {"idx": idx, "answer": "", "success": False, "error": last_error, "elapsed": elapsed, "attempt": attempt + 1}
    
    # 理论上不会到达这里，但为了安全起见
    return {"idx": idx, "answer": "", "success": False, "error": last_error or "Unknown error", "elapsed": 0, "attempt": max_retries}


def main():
    file_name = "sunny"
    parser = argparse.ArgumentParser(description="SFT model inference on test split")
    parser.add_argument("--input", default="./results/"+file_name+"/test_predictions_use_memory.csv",
                        help="Input CSV file with prompt column")
    parser.add_argument("--output", default="./results/"+file_name+"/SFT/4B_cross.csv",
                        help="Output CSV file with answer_local_LLM column added")
    parser.add_argument("--workers", type=int, default=80, help="Max parallel workers")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout seconds")
    parser.add_argument("--base-url", default="http://localhost:8003/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="./models/merge/sft_qwen3_4B", help="Served model name or local model path")
    parser.add_argument("--api-key", default="test-key", help="API key used by the local server")
    parser.add_argument("--max-tokens", type=int, default=5500, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    # 读取输入文件
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input, keep_default_na=False)
    
    if 'prompt' not in df.columns:
        print(f"Error: Input file must contain 'prompt' column. Available columns: {list(df.columns)}")
        return
    
    if 'idx' not in df.columns:
        df['idx'] = df.index
    
    print(f"Found {len(df)} rows in input file")
    
    # 检查输出文件是否存在，如果存在则加载已有结果并验证有效性
    existing_results = {}
    invalid_indices = set()  # 存储无效结果的索引
    
    if os.path.exists(args.output):
        print(f"Output file exists: {args.output}")
        try:
            df_existing = pd.read_csv(args.output, keep_default_na=False)
            if 'idx' in df_existing.columns and 'answer_local_LLM' in df_existing.columns:
                # 检查是否有ground_truth列用于验证
                has_ground_truth = 'ground_truth' in df_existing.columns
                if not has_ground_truth and 'ground_truth' in df.columns:
                    # 如果输出文件没有ground_truth，但从输入文件合并
                    df_existing = df_existing.merge(df[['idx', 'ground_truth']], on='idx', how='left')
                    has_ground_truth = True
                
                # 创建已有结果的映射，并验证有效性
                valid_count = 0
                invalid_count = 0
                for _, row in df_existing.iterrows():
                    idx = row['idx']
                    answer = str(row['answer_local_LLM']).strip()
                    # 只有当答案不为空时才检查
                    if answer and answer != '' and answer.lower() != 'nan':
                        if has_ground_truth:
                            ground_truth = str(row.get('ground_truth', '')).strip()
                            if is_valid_answer(answer, ground_truth):
                                existing_results[idx] = answer
                                valid_count += 1
                            else:
                                invalid_indices.add(idx)
                                invalid_count += 1
                        else:
                            # 没有ground_truth，无法验证，直接认为有效
                            existing_results[idx] = answer
                            valid_count += 1
                
                print(f"  Loaded {valid_count} valid existing results from output file")
                if invalid_count > 0:
                    print(f"  Found {invalid_count} invalid results that will be regenerated")
            else:
                print(f"  Warning: Output file exists but missing 'idx' or 'answer_local_LLM' column")
        except Exception as e:
            print(f"  Warning: Failed to read existing output file: {e}")
    else:
        print(f"Output file does not exist, will create new file: {args.output}")
    
    # 过滤出需要处理的行（没有已有结果的行 + 无效结果的行）
    # 合并输入文件的ground_truth到df_existing（如果存在）
    if 'ground_truth' not in df.columns:
        print("  Warning: Input file does not contain 'ground_truth' column. Cannot validate answer validity.")
    
    # 需要处理的行：没有结果的行 + 无效结果的行
    missing_indices = set(df['idx'].tolist()) - set(existing_results.keys())
    indices_to_process = missing_indices | invalid_indices
    
    df_to_process = df[df['idx'].isin(indices_to_process)].copy()
    num_to_process = len(df_to_process)
    num_skipped = len(df) - num_to_process
    
    if num_skipped > 0:
        print(f"Skipping {num_skipped} rows that already have valid results")
    if len(invalid_indices) > 0:
        print(f"Will regenerate {len(invalid_indices)} invalid results")
    print(f"Will process {num_to_process} rows")
    
    if num_to_process == 0:
        print("All rows already have results. Nothing to do.")
        return
    
    # 配置
    config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
    }
    
    print(f"Processing {num_to_process} prompts with {args.workers} workers...")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Max Tokens: {args.max_tokens}")
    
    # 并行处理
    results = existing_results.copy()  # 从已有结果开始
    import time
    start_time = time.time()
    first_result_time = None
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 检查是否有ground_truth列
        has_gt = 'ground_truth' in df_to_process.columns
        
        futures = {
            executor.submit(
                call_api, 
                row['prompt'], 
                row['idx'], 
                config, 
                args.timeout,
                max_retries=4,
                ground_truth=str(row['ground_truth']).strip() if has_gt and pd.notna(row.get('ground_truth')) else None
            ): row['idx']
            for _, row in df_to_process.iterrows()
        }
        
        print(f"Submitted {len(futures)} tasks to thread pool")
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            completed_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    # 不使用timeout，让future自然完成（因为call_api内部已经有超时控制）
                    # 如果future已经完成，result()会立即返回
                    result = future.result(timeout=None)  # 移除超时限制，因为任务已经在执行中
                    
                    if first_result_time is None:
                        first_result_time = time.time()
                        elapsed = first_result_time - start_time
                        print(f"\n[Info] First request completed in {elapsed:.1f}s")
                    
                    results[result['idx']] = result['answer']
                    completed_count += 1
                    elapsed = result.get('elapsed', 0)
                    attempt = result.get('attempt', 1)
                    if not result['success']:
                        retry_info = f" (重试 {attempt}/3)" if attempt > 1 else ""
                        print(f"\n[Error] idx={result['idx']}: {result['error']} (耗时: {elapsed:.1f}s{retry_info})")
                    elif elapsed > 30:  # 如果单个请求超过30秒，打印警告
                        retry_info = f" (重试 {attempt}/3)" if attempt > 1 else ""
                        print(f"\n[Warning] idx={result['idx']} took {elapsed:.1f}s (可能较慢{retry_info})")
                    elif attempt > 1:
                        print(f"\n[Info] idx={result['idx']} succeeded after {attempt} attempts")
                    # 静默处理成功的请求，只在每10个时打印进度
                except concurrent.futures.TimeoutError:
                    results[idx] = ""
                    print(f"\n[Error] idx={idx}: Future result timeout (任务可能仍在执行)")
                except Exception as e:
                    results[idx] = ""
                    error_type = type(e).__name__
                    print(f"\n[Error] idx={idx}: {error_type}: {e}")
                finally:
                    pbar.update(1)
                    # 每完成10个任务，打印一次进度
                    if completed_count % 10 == 0 and completed_count > 0:
                        elapsed_total = time.time() - start_time
                        print(f"[Progress] Completed {completed_count}/{len(futures)} in {elapsed_total:.1f}s")
    
    # 添加结果列（合并已有结果和新生成的结果）
    df['answer_local_LLM'] = df['idx'].map(results).fillna('')
    
    # 保存
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    
    num_success = (df['answer_local_LLM'] != '').sum()
    num_new = num_success - len(existing_results)
    print(f"Successfully generated {num_new} new answers (total: {num_success}/{len(df)} prompts)")


if __name__ == "__main__":
    main()
