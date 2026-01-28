#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行Agent对测试数据进行预测，输出格式符合sft_eval_test.py的要求
"""

import os
import sys
import json
import re
import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入tqdm，如果不存在则使用简单的进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度显示类（如果没有tqdm）
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", unit="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
            if total:
                print(f"{desc}: 0/{total} {unit}")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.total:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit} completed")
        
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit}")
        
        def set_postfix(self, **kwargs):
            pass

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入agent相关模块（顺带导入工具配置，保证工具呈现与agent一致）
from castmaster.castmaster_agent import (
    app,
    load_data_from_csv,
    format_tool_info_for_prompt,
    MANDATORY_TOOLS,
    TOOL_REGISTRY,
    TOOL_DESCRIPTIONS,
    extract_aux_model_name_from_tool_outputs,
    DATASET_NAME,
)


def run_agent_on_test_data(
    test_csv: str,
    output_csv: str,
    memory_file: str = None,
    start_idx: int = 0,
    num_samples: int = None,
    sample_idx: int = None,
    num_workers: int = 10,
):
    """
    使用Agent处理测试数据并生成符合sft_eval_test.py要求的输出
    
    Args:
        test_csv: 测试数据CSV文件路径
        output_csv: 输出CSV文件路径
        memory_file: Memory库文件路径（可选，用于test模式）
        start_idx: 起始样本索引
        num_samples: 处理的样本数量（None表示全部）
        sample_idx: 处理特定样本索引（覆盖start_idx和num_samples）
    """
    print(f"Loading test data from: {test_csv}")

    # 加载Memory库（如果提供）
    memory_library = None
    if memory_file and os.path.exists(memory_file):
        from castmaster.memory_library import MemoryLibrary
        memory_library = MemoryLibrary(similarity_threshold=0.1, use_faiss=True)
        memory_library.load(memory_file)
        print(f"Loaded Memory Library with {memory_library.size()} memories from {memory_file}")

    # 数据滑窗步长：ETTh1、MOPEX 为 48；ETTm1 及其他为 96
    _data_stride = 48 if ("ETTh1" in DATASET_NAME or "mopex" in DATASET_NAME or "MOPEX" in DATASET_NAME) else 96
    # 加载测试数据
    if sample_idx is not None:
        data_list = load_data_from_csv(test_csv, start_idx=sample_idx, num_samples=1, stride=_data_stride)
    else:
        data_list = load_data_from_csv(test_csv, start_idx=start_idx, num_samples=num_samples, stride=_data_stride)
    
    if not data_list:
        print("Error: No data found")
        return
    
    print(f"Processing {len(data_list)} samples with up to {num_workers} workers...")

    # 单个样本处理函数（用于并行执行）
    def process_one(sample_idx_local: int, sample: dict):
        # 减少详细输出，避免与进度条冲突
        # print(f"\n{'='*60}")
        # print(f"Processing sample {sample_idx_local + 1}/{len(data_list)}: idx={sample['idx']}")
        # print(f"{'='*60}\n")

        # 构建初始输入
        inputs = {
            "idx": sample['idx'],
            "input_data": sample['input_data'],
            "ground_truth_data": sample['ground_truth_data'],
            "input_json": sample['input_json'],
            "ground_truth_json": sample['ground_truth_json'],
            "memory_library": memory_library,
            "retrieved_memory": None,
            "tool": "",
            "task": "Predict the next 96 time points of daily streamflow discharge (timestamps increment by 1 day)",
            "tool_calls": [],
            "tool_outputs": {},
            "loop_count": 0,
            "mode": "test",  # 使用test模式
            "update_memory": False  # 测试模式不更新memory
        }

        try:
            final_state = app.invoke(inputs)

            # 提取结果
            parsed_answer = final_state.get("parsed_answer", "")
            tool_calls = final_state.get("tool_calls", [])
            tool_outputs = final_state.get("tool_outputs", {})
            # 修复：不能直接对DataFrame使用or操作符，需要使用is None检查
            ground_truth_data = final_state.get("ground_truth_data")
            if ground_truth_data is None:
                ground_truth_data = sample.get("ground_truth_data")

            # 构建tool信息字符串（用于tool列；与训练数据兼容）
            tool_info_str = format_tool_info_for_prompt(tool_calls, tool_outputs)

            # 提取小模型辅助预测中使用的小模型名称（如果存在）
            model_name = ""
            try:
                name_str = extract_aux_model_name_from_tool_outputs(tool_outputs)
                if isinstance(name_str, str):
                    model_name = name_str
            except Exception:
                model_name = ""

            # 直接使用 forecasting_node 中“真实传入LLM”的Prompt，保证完全一致
            full_prompt = final_state.get("forecast_full_prompt", "")
            if not full_prompt:
                # 兜底：至少记录system提示，避免prompt列为空
                full_prompt = "You are a time series forecasting expert. Your current task is to predict daily streamflow discharge (MOPEX dataset)."

            # 获取forecast_result用于提取think和answer
            forecast_result = final_state.get("forecast_result", "")
            
            # 如果parsed_answer为空，尝试从forecast_result中提取
            if not parsed_answer and forecast_result:
                # 尝试提取<answer>标签内容
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, forecast_result, re.DOTALL)
                if match:
                    parsed_answer = match.group(1).strip()

            # 提取think标签内容
            parsed_think = ""
            if forecast_result:
                pattern = r'<think>(.*?)</think>'
                match = re.search(pattern, forecast_result, re.DOTALL)
                if match:
                    parsed_think = match.group(1).strip()

            # 验证parsed_answer是否为有效的JSON
            answer_json = ""
            if parsed_answer:
                try:
                    # 尝试解析为JSON
                    answer_dict = json.loads(parsed_answer)
                    # 如果解析成功，重新序列化为JSON字符串（确保格式一致）
                    answer_json = json.dumps(answer_dict, ensure_ascii=False)
                except json.JSONDecodeError:
                    # 如果不是有效JSON，直接使用原始字符串
                    answer_json = parsed_answer
                    print(f"    >> [Warning] Answer is not valid JSON, using raw string")

            # 构建结果行（在第二列加入 model_name）
            result_row = {
                "idx": sample['idx'],
                "model_name": model_name,
                "input": sample['input_json'],
                "ground_truth": sample['ground_truth_json'],
                "tool": tool_info_str if tool_info_str else "",
                "prompt": full_prompt,
                "think": parsed_think if parsed_think else "",
                "answer": answer_json
            }

            print(f"    >> [Success] Sample {sample['idx']} processed")
            if answer_json:
                try:
                    answer_dict = json.loads(answer_json)
                    print(f"    >> [Info] Answer contains {len(answer_dict)} time points")
                except Exception:
                    print(f"    >> [Info] Answer length: {len(answer_json)} chars")
            else:
                print(f"    >> [Warning] No answer extracted for sample {sample['idx']}")

        except Exception as e:
            print(f"    >> [Error] Failed to process sample {sample['idx']}: {e}")
            import traceback
            traceback.print_exc()

            # 即使失败也记录一行（think和answer为空）
            result_row = {
                "idx": sample['idx'],
                "model_name": "",
                "input": sample['input_json'],
                "ground_truth": sample['ground_truth_json'],
                "tool": "",
                "prompt": "",
                "think": "",
                "answer": ""
            }

        # 返回（原始顺序索引, 结果行）
        return sample_idx_local, result_row

    # 并行处理所有样本
    results_ordered = [None] * len(data_list)
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        future_to_idx = {
            executor.submit(process_one, i, sample): i
            for i, sample in enumerate(data_list)
        }

        # 使用tqdm显示进度条
        with tqdm(total=len(data_list), desc="Processing samples", unit="sample") as pbar:
            for future in as_completed(future_to_idx):
                idx_local, row = future.result()
                results_ordered[idx_local] = row
                pbar.update(1)
                # 在进度条中显示当前处理的样本idx
                if row:
                    pbar.set_postfix({"current_idx": row.get("idx", "N/A")})

    # 过滤掉可能的None（理论上不应该有）
    results = [row for row in results_ordered if row is not None]

    # 保存结果到CSV
    if results:
        df_results = pd.DataFrame(results)
        
        # 确保列顺序（在第二列加入 model_name，包含think列）
        columns_order = ["idx", "model_name", "input", "ground_truth", "tool", "prompt", "think", "answer"]
        # 如果某些列不存在，只选择存在的列
        existing_columns = [col for col in columns_order if col in df_results.columns]
        df_results = df_results[existing_columns]
        
        # 保存CSV
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        df_results.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_csv}")
        print(f"Total samples processed: {len(results)}")
        print(f"{'='*60}")
    else:
        print("Error: No results to save")


def main():
    parser = argparse.ArgumentParser(description='Run Agent on test data and generate output for sft_eval_test.py')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Input test CSV path (format: date, target col e.g. OT or daily streamflow discharge, exogenous variables)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output CSV file path (will be used by sft_eval_test.py)')
    parser.add_argument('--memory_file', type=str, default=None,
                        help='Memory library file path (optional, for test mode)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for samples')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Process specific sample index (overrides start_idx and num_samples)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of parallel workers for processing samples')
    
    args = parser.parse_args()
    
    run_agent_on_test_data(
        test_csv=args.test_csv,
        output_csv=args.output_csv,
        memory_file=args.memory_file,
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        sample_idx=args.sample_idx,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
