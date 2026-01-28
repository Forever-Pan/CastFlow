#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用预测结果评测脚本
用于评测训练阶段（build_memory）和测试阶段（test）保存的预测结果
支持格式：包含 idx, answer, ground_truth 列的CSV文件
复用validate_sft.py中的calculate_metrics函数
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from typing import Dict, Tuple, List

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_json_column(value: str) -> Dict[str, float]:
    """
    解析JSON格式的列（从validate_sft.py复用）
    
    Args:
        value: JSON字符串或文本格式
        
    Returns:
        解析后的字典，key是时间戳，value是数值
    """
    if not value or not value.strip():
        return {}
    
    # 清理JSON字符串：移除可能的占位符和注释
    cleaned_value = value.strip()
    # 移除JSON中的 ... 占位符（可能是省略号）
    cleaned_value = re.sub(r',\s*\.\.\.\s*,', ',', cleaned_value)
    cleaned_value = re.sub(r',\s*\.\.\.\s*\]', ']', cleaned_value)
    cleaned_value = re.sub(r'\[\s*\.\.\.\s*,', '[', cleaned_value)
    
    # 首先尝试直接解析JSON
    try:
        parsed = json.loads(cleaned_value)
        
        # 如果解析结果是列表，尝试转换为字典
        if isinstance(parsed, list):
            # 如果是空列表，返回空字典
            if len(parsed) == 0:
                return {}
            
            # 如果列表元素是字典，尝试提取时间戳和数值
            if isinstance(parsed[0], dict):
                result = {}
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    
                    # 情况1: 字典只有一个键值对，键是时间戳，值是数值
                    if len(item) == 1:
                        timestamp = list(item.keys())[0]
                        val = list(item.values())[0]
                        try:
                            result[str(timestamp)] = float(val)
                        except (ValueError, TypeError):
                            continue
                    
                    # 情况2: 字典有多个键，尝试常见的键名
                    else:
                        timestamp = item.get('timestamp') or item.get('time') or item.get('t') or item.get('date')
                        val = item.get('value') or item.get('val') or item.get('v')
                        if timestamp and val is not None:
                            try:
                                result[str(timestamp)] = float(val)
                            except (ValueError, TypeError):
                                continue
                
                return result
            else:
                print(f"警告: JSON解析结果为列表但无法转换为字典: {type(parsed[0])}")
                return {}
        
        # 如果解析结果是字典，直接返回
        if isinstance(parsed, dict):
            return parsed
        
        # 其他类型，返回空字典
        print(f"警告: JSON解析结果为不支持的类型: {type(parsed)}")
        return {}
        
    except json.JSONDecodeError:
        # JSON解析失败，尝试从文本中提取键值对
        result = {}
        
        # 尝试匹配 "时间戳": 数值 或 时间戳: 数值 的格式
        patterns = [
            # JSON格式: "2024-11-15 00:00:00": 38.74
            r'"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"\s*:\s*([+-]?\d+\.?\d*)',
            # 文本格式: 2024-11-15 00:00:00: 38.74
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):\s*([+-]?\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, value)
            if matches:
                for timestamp, val_str in matches:
                    try:
                        result[timestamp] = float(val_str)
                    except (ValueError, TypeError):
                        continue
                if result:
                    return result
        
        # 如果所有方法都失败，返回空字典
        return {}
    
    except (ValueError, TypeError) as e:
        print(f"警告: 数据类型转换失败: {str(e)}")
        return {}


def calculate_metrics(pred_dict: Dict[str, float], gt_dict: Dict[str, float], idx: any = None) -> Tuple[float, float, float, float, float, float, int]:
    """
    计算预测值和真实值之间的各种指标（从validate_sft.py复用）
    
    Args:
        pred_dict: 预测值字典（时间戳 -> 数值）
        gt_dict: 真实值字典（时间戳 -> 数值）
        idx: 样本索引（用于报警信息）
        
    Returns:
        (MSE值, MAE值, RMSE值, 归一化MSE, 归一化MAE, 归一化RMSE, 有效数据点数量)
    """
    # 确保输入是字典类型
    if not isinstance(pred_dict, dict):
        # print(f"警告: pred_dict 不是字典类型，而是 {type(pred_dict)}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    if not isinstance(gt_dict, dict):
        # print(f"警告: gt_dict 不是字典类型，而是 {type(gt_dict)}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    if not pred_dict or not gt_dict:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    # 按照时间戳对齐数据
    common_timestamps = set(pred_dict.keys()) & set(gt_dict.keys())
    
    # 提取数值
    pred_values = []
    gt_values = []
    
    # 如果有共同的时间戳，优先使用时间戳对齐
    if common_timestamps:
        # 按照时间戳排序
        sorted_timestamps = sorted(common_timestamps)
        
        for ts in sorted_timestamps:
            try:
                pred_val = float(pred_dict[ts])
                gt_val = float(gt_dict[ts])
                pred_values.append(pred_val)
                gt_values.append(gt_val)
            except (ValueError, TypeError):
                # 跳过无效值
                continue
                
    # 如果没有共同时间戳（或非常少），但点数相同，尝试按顺序对齐（Fallback机制）
    # 用户要求：时间戳不匹配的话只需要进行一个终端输出警告，但是只要时间点格式匹配就也进行mse和mae的计算
    elif len(pred_dict) == len(gt_dict) and len(pred_dict) > 0:
        if idx is not None:
            print(f"警告: 样本 {idx} 时间戳完全不匹配，但数据点数量一致 ({len(pred_dict)})，采用按顺序强制对齐计算指标。")
            pred_ts_sample = sorted(list(pred_dict.keys()))[:3]
            gt_ts_sample = sorted(list(gt_dict.keys()))[:3]
            print(f"  - Answer时间戳示例: {pred_ts_sample}")
            print(f"  - GT时间戳示例: {gt_ts_sample}")
            
        # 按时间戳排序后取值
        sorted_pred_ts = sorted(pred_dict.keys())
        sorted_gt_ts = sorted(gt_dict.keys())
        
        for pt, gt in zip(sorted_pred_ts, sorted_gt_ts):
            try:
                pred_val = float(pred_dict[pt])
                gt_val = float(gt_dict[gt])
                pred_values.append(pred_val)
                gt_values.append(gt_val)
            except (ValueError, TypeError):
                continue
                
    else:
        # 既没有共同时间戳，点数也不一致，无法计算
        pred_ts_sample = sorted(list(pred_dict.keys()))[:3] if pred_dict else []
        gt_ts_sample = sorted(list(gt_dict.keys()))[:3] if gt_dict else []
        if pred_ts_sample and gt_ts_sample:
            # 只有在确实无法计算时才报错
            print(f"警告: 样本 {idx if idx is not None else '?'} 时间戳不匹配且数量不一致 (Answer:{len(pred_dict)}, GT:{len(gt_dict)}) - Answer示例: {pred_ts_sample}, GT示例: {gt_ts_sample}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    if len(pred_values) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    # 计算指标
    pred_array = np.array(pred_values)
    gt_array = np.array(gt_values)
    
    mse = np.mean((pred_array - gt_array) ** 2)
    mae = np.mean(np.abs(pred_array - gt_array))
    rmse = np.sqrt(mse)
    
    # 计算归一化指标
    # 使用ground_truth的统计量进行归一化
    gt_mean = np.mean(gt_array)
    gt_std = np.std(gt_array)
    gt_var = np.var(gt_array)
    
    # 归一化MSE: 使用方差归一化 (MSE / Var(gt))
    if gt_var > 0:
        nmse = mse / gt_var
    else:
        nmse = float('nan')
    
    # 归一化MAE: 使用均值归一化 (MAE / mean(gt))
    if abs(gt_mean) > 1e-10:
        nmae = mae / abs(gt_mean)
    else:
        nmae = float('nan')
    
    # 归一化RMSE: 使用均值归一化 (RMSE / mean(gt))
    if abs(gt_mean) > 1e-10:
        nrmse = rmse / abs(gt_mean)
    else:
        nrmse = float('nan')
    
    return mse, mae, rmse, nmse, nmae, nrmse, len(pred_values)


def plot_mae_mse_distribution(
    sample_indices: List[int],
    mse_values: List[float],
    mae_values: List[float],
    output_path: str,
    mode_str: str = "预测结果"
):
    """
    生成MAE vs MSE的二维分布图
    
    Args:
        sample_indices: 样本索引列表
        mse_values: MSE值列表
        mae_values: MAE值列表
        output_path: 输出图像文件路径
        mode_str: 模式字符串（用于标题）
    """
    if len(sample_indices) == 0 or len(mse_values) == 0 or len(mae_values) == 0:
        print("警告: 没有有效数据可以绘制图表")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    scatter = plt.scatter(mse_values, mae_values, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidths=0.5)
    
    # 标注所有点的索引（用户要求能看到所有点的标签）
    # 使用较小的字体和简单的样式，避免图表过于拥挤
    for i, (mse, mae, idx) in enumerate(zip(mse_values, mae_values, sample_indices)):
        plt.annotate(
            str(idx),
            (mse, mae),
            fontsize=6,
            alpha=0.7,
            xytext=(2, 2),
            textcoords='offset points',
            bbox=dict(
                boxstyle='round,pad=0.2',
                facecolor='yellow',
                alpha=0.3,
                edgecolor='none'
            )
        )
    
    # 设置标签和标题（使用英文避免字体问题，或使用支持中文的字体）
    plt.xlabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
    plt.ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    plt.title(f'{mode_str} - MAE vs MSE Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息文本框（改进样式，使其更清晰）
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    median_mse = np.median(mse_values)
    median_mae = np.median(mae_values)
    std_mse = np.std(mse_values)
    std_mae = np.std(mae_values)
    
    # 使用英文避免字体问题
    stats_text = f'Samples: {len(sample_indices)}\n'
    stats_text += f'Avg MSE: {avg_mse:.4f} (Median: {median_mse:.4f})\n'
    stats_text += f'Avg MAE: {avg_mae:.4f} (Median: {median_mae:.4f})\n'
    stats_text += f'Std Dev: MSE={std_mse:.4f}, MAE={std_mae:.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             family='monospace',  # 使用等宽字体，使数字对齐
             bbox=dict(
                 boxstyle='round,pad=0.5',
                 facecolor='white',
                 alpha=0.85,
                 edgecolor='black',
                 linewidth=1.2
             ))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_path}")
    
    # 关闭图形以释放内存
    plt.close()


def plot_worst_samples_series(
    df: pd.DataFrame,
    answer_col: str,
    gt_col: str,
    mse_list: List[float],
    row_indices: List[int],
    output_dir: str,
    top_k: int = 15,
) -> None:
    """
    根据 MSE 从大到小选出若干样本，绘制真实序列与预测序列两条曲线。

    Args:
        df: 原始 DataFrame（包含 answer / ground_truth 列）
        answer_col: 预测列名
        gt_col: 真实值列名
        mse_list: 每个有效样本对应的 MSE 列表（顺序与 row_indices 对应）
        row_indices: 有效样本在 df 中的行索引列表
        output_dir: 输出图片目录
        top_k: 需要展示的样本数量（默认 15）
    """
    if not mse_list or not row_indices:
        print("警告: 没有有效样本用于绘制时间序列曲线")
        return

    # 选出 MSE 最大的 top_k 个样本（基于 mse_list 的顺序）
    num_samples = len(mse_list)
    k = min(top_k, num_samples)
    # indices_in_valid_list 是 mse_list 中的下标
    sorted_indices = sorted(range(num_samples), key=lambda i: mse_list[i], reverse=True)[:k]

    os.makedirs(output_dir, exist_ok=True)

    for rank, valid_idx in enumerate(sorted_indices, start=1):
        row_idx = row_indices[valid_idx]
        mse_val = mse_list[valid_idx]

        row = df.loc[row_idx]
        idx_val = row.get("idx", row_idx)

        answer_str = str(row.get(answer_col, ""))
        gt_str = str(row.get(gt_col, ""))

        pred_dict = parse_json_column(answer_str)
        gt_dict = parse_json_column(gt_str)

        if not isinstance(pred_dict, dict) or not isinstance(gt_dict, dict) or not pred_dict or not gt_dict:
            print(f"    >> [Series Plot] 样本 idx={idx_val} 的预测或真实序列解析失败，跳过")
            continue

        # 对齐时间戳并排序
        common_ts = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
        if not common_ts:
            print(f"    >> [Series Plot] 样本 idx={idx_val} 的时间戳不匹配，跳过")
            continue

        # 转为列表用于绘图
        x = pd.to_datetime(common_ts)
        y_pred = [float(pred_dict[t]) for t in common_ts]
        y_gt = [float(gt_dict[t]) for t in common_ts]

        plt.figure(figsize=(12, 6))
        plt.plot(x, y_gt, label="Ground Truth", color="tab:blue", linewidth=2)
        plt.plot(x, y_pred, label="Prediction", color="tab:orange", linewidth=2, linestyle="--")

        # 尝试从 df 中获取模型信息（如果存在相关列）
        model_name = None
        for key in ["model", "model_name", "best_model"]:
            if key in df.columns:
                val = row.get(key)
                if isinstance(val, str) and val.strip():
                    model_name = val.strip()
                    break

        title_parts = [f"Sample idx={idx_val}", f"MSE={mse_val:.4f}"]
        if model_name:
            title_parts.append(f"Model={model_name}")
        title = " | ".join(title_parts)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"sample_idx_{idx_val}_mse_{mse_val:.4f}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    >> [Series Plot] 已保存样本 idx={idx_val} 的序列图: {out_path}")


def evaluate_predictions(csv_file: str, output_csv: str = None, mode: str = "auto"):
    """
    评测预测结果（支持训练和测试阶段）
    
    Args:
        csv_file: 输入CSV文件路径（包含idx, answer, ground_truth列，或run_agent_test.py的输出格式）
        output_csv: 输出CSV文件路径（可选，包含评测指标）
        mode: 模式标识（"train", "test", "auto"），用于显示不同的标题
    """
    print(f"正在读取文件: {csv_file}")
    df = pd.read_csv(csv_file, keep_default_na=False)
    
    print(f"总共有 {len(df)} 行数据")
    
    # 检查必需的列（支持两种格式）
    # 格式1: idx, answer, ground_truth (训练阶段保存的格式)
    # 格式2: idx, input, ground_truth, tool, prompt, think, answer (run_agent_test.py的输出格式)
    if 'answer' in df.columns and 'ground_truth' in df.columns:
        answer_col = 'answer'
        gt_col = 'ground_truth'
    elif 'ground_truth' in df.columns:
        # 尝试查找answer列的其他可能名称
        possible_answer_cols = [col for col in df.columns if 'answer' in col.lower() or 'pred' in col.lower()]
        if possible_answer_cols:
            answer_col = possible_answer_cols[0]
            gt_col = 'ground_truth'
            print(f"使用列 '{answer_col}' 作为预测结果列")
        else:
            print(f"❌ 错误: 找不到answer列，可用列: {list(df.columns)}")
            return
    else:
        print(f"❌ 错误: 缺少必需的列")
        print(f"   文件应包含 'answer' 和 'ground_truth' 列")
        print(f"   当前列: {list(df.columns)}")
        return
    
    # 自动检测模式
    if mode == "auto":
        if 'input' in df.columns and 'tool' in df.columns:
            mode = "test"
        else:
            mode = "train"
    
    mode_str = "测试阶段" if mode == "test" else "训练阶段"
    
    # 统计信息
    valid_samples = 0
    invalid_samples = 0
    mse_list = []
    mae_list = []
    rmse_list = []
    nmse_list = []  # 归一化MSE
    nmae_list = []  # 归一化MAE
    nrmse_list = []  # 归一化RMSE
    total_points = 0
    # 用于可视化的数据
    sample_indices = []  # 存储每个有效样本的索引
    row_indices: List[int] = []  # 存储每个有效样本在 df 中的行索引
    
    # 处理每一行
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        
        # 解析answer和ground_truth
        answer_str = str(row.get(answer_col, ''))
        ground_truth_str = str(row.get(gt_col, ''))
        
        answer_dict = parse_json_column(answer_str)
        gt_dict = parse_json_column(ground_truth_str)
        
        # 计算指标（包括归一化指标）
        mse, mae, rmse, nmse, nmae, nrmse, num_points = calculate_metrics(answer_dict, gt_dict, idx=idx_val)
        
        if np.isnan(mse):
            invalid_samples += 1
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 行... (无效: {invalid_samples})")
            continue
        
        valid_samples += 1
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        sample_indices.append(idx_val)  # 保存样本索引用于可视化
        row_indices.append(idx)  # 保存 DataFrame 行索引用于后续绘图
        if not np.isnan(nmse):
            nmse_list.append(nmse)
        if not np.isnan(nmae):
            nmae_list.append(nmae)
        if not np.isnan(nrmse):
            nrmse_list.append(nrmse)
        total_points += num_points
        
        # 显示进度
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 行... (有效: {valid_samples}, 无效: {invalid_samples})")
    
    # 计算平均指标
    if len(mse_list) == 0:
        print("❌ 错误: 没有有效的样本可以计算指标")
        return
    
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    
    # 计算归一化指标的平均值
    avg_nmse = np.mean(nmse_list) if len(nmse_list) > 0 else float('nan')
    avg_nmae = np.mean(nmae_list) if len(nmae_list) > 0 else float('nan')
    avg_nrmse = np.mean(nrmse_list) if len(nrmse_list) > 0 else float('nan')
    
    # 计算其他统计信息
    min_mse = np.min(mse_list)
    max_mse = np.max(mse_list)
    median_mse = np.median(mse_list)
    std_mse = np.std(mse_list)
    
    median_mae = np.median(mae_list)
    median_rmse = np.median(rmse_list)
    
    median_nmse = np.median(nmse_list) if len(nmse_list) > 0 else float('nan')
    median_nmae = np.median(nmae_list) if len(nmae_list) > 0 else float('nan')
    median_nrmse = np.median(nrmse_list) if len(nrmse_list) > 0 else float('nan')
    
    # 输出结果
    print("\n" + "=" * 60)
    print(f"{mode_str}预测结果评测统计")
    print("=" * 60)
    print(f"总样本数: {len(df)}")
    print(f"有效样本数: {valid_samples}")
    print(f"无效样本数: {invalid_samples}")
    print(f"总数据点数: {total_points}")
    print(f"平均每个样本数据点数: {total_points / valid_samples:.2f}" if valid_samples > 0 else "")
    print("\nMSE (均方误差) 统计:")
    print(f"  平均MSE: {avg_mse:.4f}")
    print(f"  中位数MSE: {median_mse:.4f}")
    print(f"  最小MSE: {min_mse:.4f}")
    print(f"  最大MSE: {max_mse:.4f}")
    print(f"  标准差: {std_mse:.4f}")
    
    print(f"\nMAE (平均绝对误差) 统计:")
    print(f"  平均MAE: {avg_mae:.4f}")
    print(f"  中位数MAE: {median_mae:.4f}")
    
    print(f"\nRMSE (均方根误差) 统计:")
    print(f"  平均RMSE: {avg_rmse:.4f}")
    print(f"  中位数RMSE: {median_rmse:.4f}")
    
    print(f"\n归一化指标统计:")
    print(f"  归一化MSE (NMSE): {avg_nmse:.4f} (中位数: {median_nmse:.4f})")
    print(f"  归一化MAE (NMAE): {avg_nmae:.4f} (中位数: {median_nmae:.4f})")
    print(f"  归一化RMSE (NRMSE): {avg_nrmse:.4f} (中位数: {median_nrmse:.4f})")
    print("=" * 60)
    
    # 生成可视化图表
    # 确定输出图像路径和时间序列图路径（windy/sunny/ETTm1/ETTh1/EPF_NP/EPF_BE/EPF_DE/EPF_FR/EPF_PJM 单独目录；其余用 results/pic_analysis）
    if "windy" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "windy", "pic_analysis")
    elif "sunny" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "sunny", "pic_analysis")
    elif "ETTm1" in csv_file or "ETT_ETTm1" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "ETTm1", "pic_analysis")
    elif "mopex" in csv_file or "MOPEX" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "MOPEX", "pic_analysis")
    elif "ETTh1" in csv_file or "ETT_ETTh1" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "ETTh1", "pic_analysis")
    elif "EPF_NP" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "EPF_NP", "pic_analysis")
    elif "EPF_BE" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "EPF_BE", "pic_analysis")
    elif "EPF_DE" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "EPF_DE", "pic_analysis")
    elif "EPF_FR" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "EPF_FR", "pic_analysis")
    elif "EPF_PJM" in csv_file:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "EPF_PJM", "pic_analysis")
    else:
        pic_base_dir = os.path.join(PROJECT_ROOT, "results", "pic_analysis")
    os.makedirs(pic_base_dir, exist_ok=True)
    
    # 从输入文件名生成输出图像文件名
    input_basename = os.path.splitext(os.path.basename(csv_file))[0]
    output_image_path = os.path.join(pic_base_dir, f"{input_basename}_mae_mse_distribution.png")
    
    # 生成MAE vs MSE分布图
    plot_mae_mse_distribution(
        sample_indices=sample_indices,
        mse_values=mse_list,
        mae_values=mae_list,
        output_path=output_image_path,
        mode_str=mode_str
    )

    # 生成 MSE 最大的若干样本的真实 / 预测时间序列对比图
    series_output_dir = os.path.join(pic_base_dir, f"{input_basename}_worst_series")
    plot_worst_samples_series(
        df=df,
        answer_col=answer_col,
        gt_col=gt_col,
        mse_list=mse_list,
        row_indices=row_indices,
        output_dir=series_output_dir,
        top_k=15,
    )
    
    # 保存详细结果到CSV（如果指定了输出文件）
    if output_csv:
        result_df = df.copy()
        result_df['mse'] = None
        result_df['mae'] = None
        result_df['rmse'] = None
        result_df['nmse'] = None  # 归一化MSE
        result_df['nmae'] = None  # 归一化MAE
        result_df['nrmse'] = None  # 归一化RMSE
        result_df['num_points'] = 0
        
        for idx, row in df.iterrows():
            answer_str = str(row.get(answer_col, ''))
            ground_truth_str = str(row.get(gt_col, ''))
            answer_dict = parse_json_column(answer_str)
            gt_dict = parse_json_column(ground_truth_str)
            mse, mae, rmse, nmse, nmae, nrmse, num_points = calculate_metrics(answer_dict, gt_dict, idx=idx)
            
            if not np.isnan(mse):
                result_df.at[idx, 'mse'] = mse
                result_df.at[idx, 'mae'] = mae
                result_df.at[idx, 'rmse'] = rmse
                result_df.at[idx, 'nmse'] = nmse if not np.isnan(nmse) else None
                result_df.at[idx, 'nmae'] = nmae if not np.isnan(nmae) else None
                result_df.at[idx, 'nrmse'] = nrmse if not np.isnan(nrmse) else None
                result_df.at[idx, 'num_points'] = num_points
        
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        result_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n详细结果已保存到: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions (supports both training and test phases)')
    parser.add_argument('--predictions_csv', type=str, required=True,
                        help='Input CSV file path containing predictions (format: idx, answer, ground_truth, or run_agent_test.py output format)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file path with evaluation metrics (optional)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'auto'], default='auto',
                        help='Mode identifier (train/test/auto). Auto-detect if not specified.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions_csv):
        print(f"❌ 错误: 文件不存在: {args.predictions_csv}")
        return
    
    evaluate_predictions(args.predictions_csv, args.output_csv, args.mode)


if __name__ == "__main__":
    main()
