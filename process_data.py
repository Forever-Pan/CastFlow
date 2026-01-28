#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本（合并版）
直接从源文件windy_power_train.csv生成包含idx, input, ground_truth, tool, prompt五个属性的CSV文件
使用滑动窗口，每次步进96个点，可以生成更多样本
"""

import pandas as pd
import json
import sys

def process_data(input_file, output_file, look_back=96, ground_truth_length=96, stride=None):
    """
    读取源CSV文件，进行切片并添加prompt和tool列，输出包含5个属性的CSV
    使用滑动窗口方式生成样本
    
    Args:
        input_file: 输入CSV文件路径（源文件）
        output_file: 输出CSV文件路径
        look_back: 历史窗口长度（input 点数），默认 96
        ground_truth_length: ground_truth 长度（预测点数），默认 96；EPF-NP 等用 24
        stride: 滑窗步长，None 时等于 look_back
    """
    stride = stride if stride is not None else look_back

    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 确保数据按date排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"总共有 {len(df)} 行数据 (look_back={look_back}, ground_truth_length={ground_truth_length}, stride={stride})")
    
    # 准备输出数据
    output_data = []
    
    idx = 0
    i = 0
    
    # 滑动窗口：直到没有足够的 look_back + ground_truth_length
    while i + look_back + ground_truth_length <= len(df):
        # 获取 input 数据（look_back 个点）
        input_data = df.iloc[i:i+look_back]
        
        # 获取 ground_truth 数据（接下来 ground_truth_length 个点）
        ground_truth_data = df.iloc[i+look_back:i+look_back+ground_truth_length]
        
        # 构建input的JSON：key是date，value是real_power
        input_dict = {}
        for _, row in input_data.iterrows():
            input_dict[str(row['date'])] = float(row['real_power'])
        
        # 构建ground_truth的JSON：key是date，value是real_power
        ground_truth_dict = {}
        for _, row in ground_truth_data.iterrows():
            ground_truth_dict[str(row['date'])] = float(row['real_power'])
        
        # 转换为JSON字符串
        input_json = json.dumps(input_dict, ensure_ascii=False)
        ground_truth_json = json.dumps(ground_truth_dict, ensure_ascii=False)
        
        # tool列，目前为空字符串，后续会使用
        tool = ""  # memory(input) -> (tool_call,tool_result)
        domain_description = ""  # Todo
        
        # 格式化prompt
        prompt = f"""
You are a time series forecasting expert. 
Your current task is to predict power generation in a wind power scenario. 
{domain_description}
Please predict the upcoming {ground_truth_length} time points based on the historical time series of {look_back} time points. 
The timestamps increment by 15 minutes. 
The current input segment of the historical time series is as follows: {input_json}. 
The tool invocation type and tool output results are as follows: {tool}. 
Please think before you answer. Place your thinking process inside <think></think>. 
During the thinking process, you need to consider the information provided by the tools (if any); 
place this specific information under <tool_think></tool_think> within the thinking process. 
Place the results inside <answer></answer>; the format must contain exactly {ground_truth_length} time points.
        """.strip()

# windy
# 你是一个时间序列预测专家。当前任务是需要你预测风力发电场景下的发电功率。
# {domain_description}
# 请你根据历史时间序列的96个时间点,预测接下来的96个时间点。
# 时间戳以15分钟递增。
# 当前的历史时间序列片段输入如下:{input_json}。
# 其工具调用的类型与工具输出结果如下:{tool}。
# 请你先进行思考，再回答。
# 思考过程放到<think></think>中。
# 在思考的过程中需要考虑到工具提供的信息(如果有),
# 把这部分信息放到思考过程中的<tool_think></tool_think>下。
# 结果放到<answer></answer>中，结果的格式与输入的96个点一致。 
# You are a time series forecasting expert. 
# Your current task is to predict power generation in a wind power scenario. 
# {domain_description}
# Please predict the upcoming 96 time points based on the historical time series of 96 time points. 
# The timestamps increment by 15 minutes. 
# The current input segment of the historical time series is as follows: {input_json}. 
# The tool invocation type and tool output results are as follows: {tool}. 
# Please think before you answer. Place your thinking process inside <think></think>. 
# During the thinking process, you need to consider the information provided by the tools (if any); 
# place this specific information under <tool_think></tool_think> within the thinking process. 
# Place the results inside <answer></answer>; the format of the results must be consistent with the input 96 points.

# EPF-NP
# 你是一个时间序列预测专家。当前任务是需要你预测风力发电场景下的发电功率。
# {domain_description}
# 请你根据历史时间序列的96个时间点,预测接下来的96个时间点。
# 时间戳以15分钟递增。
# 当前的历史时间序列片段输入如下:{input_json}。
# 其工具调用的类型与工具输出结果如下:{tool}。
# 请你先进行思考，再回答。
# 思考过程放到<think></think>中。
# 在思考的过程中需要考虑到工具提供的信息(如果有),
# 把这部分信息放到思考过程中的<tool_think></tool_think>下。
# 结果放到<answer></answer>中，结果的格式与输入的96个点一致。 

        
        # 添加到输出数据
        output_data.append({
            'idx': idx,
            'input': input_json,
            'ground_truth': ground_truth_json,
            'tool': tool,
            'prompt': prompt
        })
        
        idx += 1
        i += stride
        
        # 显示进度
        if idx % 10 == 0:
            print(f"已处理 {idx} 个样本...")
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(output_data)
    
    # 确保列的顺序：idx, input, ground_truth, tool, prompt
    output_df = output_df[['idx', 'input', 'ground_truth', 'tool', 'prompt']]
    
    # 保存到CSV文件
    print(f"正在保存到: {output_file}")
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"完成！共生成 {len(output_df)} 个样本")
    print(f"输出文件: {output_file}")
    
    # 显示第一行的prompt示例
    print("\n第一行prompt示例（前200个字符）：")
    print(output_df.iloc[0]['prompt'][:200] + "...")

if __name__ == "__main__":
    input_file = "datasets/windy_power_test.csv"
    output_file = "datasets/windy_power_processed_test.csv"
    look_back = 96
    ground_truth_length = 96

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        look_back = int(sys.argv[3])
    if len(sys.argv) > 4:
        ground_truth_length = int(sys.argv[4])

    process_data(input_file, output_file, look_back=look_back, ground_truth_length=ground_truth_length)

