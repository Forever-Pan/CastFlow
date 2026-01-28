#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 results/windy/test_evaluation_no_memory_final.csv
转换为可用于 SFT 训练的 CSV：
只保留 prompt 和 full_answer，重命名 full_answer -> response，
输出到 datasets/SFT_RL_bank/windy/grok_no_memory.csv
"""

import os
import pandas as pd


ROOT_DIR = "/data/Forever_Pan/AGI_sources/CastMaster_new"

SRC_PATH = os.path.join(
    ROOT_DIR,
    "results",
    "FR",
    "train_predictions_no_memory.csv",
)

DST_DIR = os.path.join(
    ROOT_DIR,
    "datasets",
    "SFT_RL_bank",
    "FR",
)

DST_PATH = os.path.join(DST_DIR, "grok_no_memory.csv")


def convert_eval_to_sft(src_path: str, dst_path: str) -> None:
    """读取评估结果 CSV，转换为 SFT 所需的 prompt/response 格式"""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"找不到源文件: {src_path}")

    print(f"读取: {src_path}")
    df = pd.read_csv(src_path)

    required_cols = ["prompt", "full_answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"源文件缺少列: {missing}，实际列为: {list(df.columns)}")

    # 构造 SFT 需要的两列：prompt + response
    sft_df = df[["prompt", "full_answer"]].rename(columns={"full_answer": "response"})

    # 删除 response 为空的行
    sft_df["response"] = sft_df["response"].astype(str)
    before = len(sft_df)
    sft_df = sft_df[sft_df["response"].str.strip() != ""]
    after = len(sft_df)

    if after == 0:
        raise ValueError("过滤后没有有效样本，请检查源数据的 full_answer 列")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sft_df.to_csv(dst_path, index=False)

    print(f"已保存到: {dst_path}")
    print(f"原始样本数: {before}，过滤后样本数: {after}")


if __name__ == "__main__":
    convert_eval_to_sft(SRC_PATH, DST_PATH)


