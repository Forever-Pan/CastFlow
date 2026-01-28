#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT验证脚本
根据sft_input.csv计算answer和ground_truth之间的平均MSE
"""

import pandas as pd
import json
import sys
import re
import numpy as np
from typing import Dict, List, Tuple

def parse_json_column(value: str) -> Dict[str, float]:
    """
    解析JSON格式的列
    
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
                    # 例如: [{"2024-11-13 18:00:00": 18.04}, ...]
                    if len(item) == 1:
                        timestamp = list(item.keys())[0]
                        val = list(item.values())[0]
                        try:
                            result[str(timestamp)] = float(val)
                        except (ValueError, TypeError):
                            continue
                    
                    # 情况2: 字典有多个键，尝试常见的键名
                    # 例如: [{"timestamp": "2024-11-12 00:00:00", "value": 4.84}, ...]
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
                # 如果列表元素不是字典，无法转换，返回空字典
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
        # 格式可能是: "2024-11-15 32:15:00: 38.74" 或 "{'2024-11-15 32:15:00': 38.74}"
        result = {}
        
        # 尝试匹配 "时间戳": 数值 或 时间戳: 数值 的格式
        # 匹配模式: "YYYY-MM-DD HH:MM:SS": 数值 或 YYYY-MM-DD HH:MM:SS: 数值
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

def calculate_metrics(pred_dict: Dict[str, float], gt_dict: Dict[str, float]) -> Tuple[float, float, float, float, float, float, int]:
    """
    计算预测值和真实值之间的各种指标（包括归一化指标）
    
    Args:
        pred_dict: 预测值字典（时间戳 -> 数值）
        gt_dict: 真实值字典（时间戳 -> 数值）
        
    Returns:
        (MSE值, MAE值, RMSE值, 归一化MSE, 归一化MAE, 归一化RMSE, 有效数据点数量)
    """
    # 确保输入是字典类型
    if not isinstance(pred_dict, dict):
        print(f"警告: pred_dict 不是字典类型，而是 {type(pred_dict)}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    if not isinstance(gt_dict, dict):
        print(f"警告: gt_dict 不是字典类型，而是 {type(gt_dict)}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    if not pred_dict or not gt_dict:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0
    
    # 按照时间戳对齐数据
    common_timestamps = set(pred_dict.keys()) & set(gt_dict.keys())
    
    # 修改：如果时间戳不匹配，尝试按索引对齐 (Index-based Fallback)
    pred_values = []
    gt_values = []
    
    if not common_timestamps:
        pred_ts_sorted = sorted(list(pred_dict.keys()))
        gt_ts_sorted = sorted(list(gt_dict.keys()))
        
        # 只有当两者都有数据时才尝试回退策略
        if pred_ts_sorted and gt_ts_sorted:
            # print(f"警告: 时间戳不匹配，尝试按顺序对齐。 Pred len={len(pred_ts_sorted)}, GT len={len(gt_ts_sorted)}")
            
            # 取较短的长度
            min_len = min(len(pred_ts_sorted), len(gt_ts_sorted))
            
            for i in range(min_len):
                try:
                    p_val = float(pred_dict[pred_ts_sorted[i]])
                    g_val = float(gt_dict[gt_ts_sorted[i]])
                    pred_values.append(p_val)
                    gt_values.append(g_val)
                except:
                    continue
    else:
        # 正常时间戳对齐逻辑
        sorted_timestamps = sorted(common_timestamps)
        for ts in sorted_timestamps:
            try:
                pred_val = float(pred_dict[ts])
                gt_val = float(gt_dict[ts])
                pred_values.append(pred_val)
                gt_values.append(gt_val)
            except (ValueError, TypeError):
                continue
    
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
    gt_range = np.max(gt_array) - np.min(gt_array)
    
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

def validate_sft(csv_file: str):
    """
    验证SFT结果，计算平均MSE
    
    Args:
        csv_file: 输入CSV文件路径
    """
    print(f"正在读取文件: {csv_file}")
    df = pd.read_csv(csv_file, keep_default_na=False)
    
    print(f"总共有 {len(df)} 行数据")
    
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
    
    # 处理每一行
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        
        # 解析answer和ground_truth
        answer_str = str(row.get('answer', ''))
        ground_truth_str = str(row.get('ground_truth', ''))
        
        answer_dict = parse_json_column(answer_str)
        gt_dict = parse_json_column(ground_truth_str)
        
        # 计算指标（包括归一化指标）
        mse, mae, rmse, nmse, nmae, nrmse, num_points = calculate_metrics(answer_dict, gt_dict)
        
        if np.isnan(mse):
            invalid_samples += 1
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 行... (无效: {invalid_samples})")
            continue
        
        valid_samples += 1
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
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
    # 使用样本标准差（ddof=1）而不是总体标准差（ddof=0）
    std_mse = np.std(mse_list, ddof=1) if len(mse_list) > 1 else 0.0
    
    median_mae = np.median(mae_list)
    median_rmse = np.median(rmse_list)
    
    median_nmse = np.median(nmse_list) if len(nmse_list) > 0 else float('nan')
    median_nmae = np.median(nmae_list) if len(nmae_list) > 0 else float('nan')
    median_nrmse = np.median(nrmse_list) if len(nrmse_list) > 0 else float('nan')
    
    # 输出结果
    print("\n" + "=" * 60)
    print("验证结果统计")
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
    
    # 保存详细结果到CSV
    result_df = df.copy()
    result_df['mse'] = None
    result_df['mae'] = None
    result_df['rmse'] = None
    result_df['nmse'] = None  # 归一化MSE
    result_df['nmae'] = None  # 归一化MAE
    result_df['nrmse'] = None  # 归一化RMSE
    result_df['num_points'] = 0
    
    for idx, row in df.iterrows():
        answer_str = str(row.get('answer', ''))
        ground_truth_str = str(row.get('ground_truth', ''))
        answer_dict = parse_json_column(answer_str)
        gt_dict = parse_json_column(ground_truth_str)
        mse, mae, rmse, nmse, nmae, nrmse, num_points = calculate_metrics(answer_dict, gt_dict)
        
        if not np.isnan(mse):
            result_df.at[idx, 'mse'] = mse
            result_df.at[idx, 'mae'] = mae
            result_df.at[idx, 'rmse'] = rmse
            result_df.at[idx, 'nmse'] = nmse if not np.isnan(nmse) else None
            result_df.at[idx, 'nmae'] = nmae if not np.isnan(nmae) else None
            result_df.at[idx, 'nrmse'] = nrmse if not np.isnan(nrmse) else None
            result_df.at[idx, 'num_points'] = num_points
    
    output_file = csv_file.replace('.csv', '_validation.csv')
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    csv_file = "./datasets/qwen_0.6B.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"输入文件: {csv_file}")
    print("-" * 60)
    
    validate_sft(csv_file)

