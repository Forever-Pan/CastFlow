# -*- coding: utf-8 -*-
"""
简化的训练脚本：使用 LiteAgent 进行训练
pkill -f "python train.py"
pkill -f AgentLightning-AgentOpsServer
ray stop --force
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import agentlightning as agl
from agentlightning.adapter import TracerTraceToTriplet

from rl_agent import LiteAgent

from pathlib import Path
import shutil

# 修复CUDA_HOME环境变量（避免DeepSpeed导入错误）
# 检查并设置正确的CUDA_HOME路径
nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = str(Path(nvcc_path).parent.parent)
    os.environ["CUDA_HOME"] = cuda_home
else:
    # 如果nvcc不在PATH中，尝试常见的CUDA路径
    common_cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11.8",
    ]
    for cuda_path in common_cuda_paths:
        nvcc_file = Path(cuda_path) / "bin" / "nvcc"
        if nvcc_file.exists():
            os.environ["CUDA_HOME"] = cuda_path
            # 同时添加到PATH以便后续使用
            os.environ["PATH"] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
            break

# 清理CUDA_HOME中的错误格式（移除开头的冒号等）
if "CUDA_HOME" in os.environ:
    cuda_home = os.environ["CUDA_HOME"].strip()
    # 移除开头的冒号
    if cuda_home.startswith(":"):
        cuda_home = cuda_home[1:]
    # 如果有多个路径（用冒号分隔），取第一个有效路径
    if ":" in cuda_home:
        cuda_home = cuda_home.split(":")[0]
    os.environ["CUDA_HOME"] = cuda_home

if "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_" + os.getlogin()

os.environ["WANDB_MODE"] = "offline"

def get_training_config(dataset_path: str, model_path: str, output_dir: str) -> Dict[str, Any]:
    """获取训练配置（已根据最新指令调整梯度范数与 KL 约束）"""
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": True, 
            "kl_coeff": 0.0,  # 💡 调低/关闭奖励中的 KL 惩罚系数
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.0, # 💡 指令要求：完全关闭 KL 控制
            }
        },
        "data": {
            "train_files": dataset_path,
            "val_files": dataset_path,
            "train_batch_size": 8,
            "max_prompt_length": 10000,
            "max_response_length": 5000,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "rollout": {
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 2,
                "name": "vllm",
                "gpu_memory_utilization": 0.65,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 35000,
                "enable_chunked_prefill": True,
                "sampling_params": {
                    "temperature": 1.3,  # 提高温度增加多样性（从1.0提高到1.3）
                    "top_p": 0.85,       # 降低top_p增加采样范围（从0.9降到0.85）
                    "top_k": 100,        # 降低top_k增加候选token多样性（从150降到100）
                    "repetition_penalty": 1.05,  # 添加重复惩罚，避免过度重复reference_prediction
                },
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes",
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 2,
                "use_kl_loss": False,      # 💡 指令要求：关闭 KL Loss
                "kl_loss_coef": 0.0,      # 💡 指令要求：系数设为 0
                "grad_clip": 2.0,         # 💡 指令要求：调大梯度裁剪阈值
                "clip_ratio_low": 0.2,    # 💡 PPO 裁剪下限
                "clip_ratio_high": 0.28,  # 💡 PPO 裁剪上限
                "clip_ratio_c": 10.0,     # 💡 较大的裁剪常数，减少约束
                "optim": {
                    "lr": 2e-6,
                    "weight_decay": 0.01
                },
                "checkpoint": {
                    "save_contents": ["hf_model"],
                    "load_contents": ["hf_model"],
                    "async_save": False,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 2,
            },
            "model": {
                "path": model_path,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": False,
            "logger": ["console", "wandb"],
            "project_name": "VLMTimeSSeriesAgent",
            "experiment_name": "vlm_tsss_training",
            "nnodes": 1,
            "test_freq": 1000,
            "total_epochs": 3,
            "save_freq": 280,
            "default_local_dir": output_dir,
        },
    }
## /data/wuli_error/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

def train_fine_grained(dataset_path: str, model_path: str, output_dir: str, port: int):
    """
    使用 LiteAgent 进行训练
    
    Args:
        dataset_path: 数据集路径（parquet文件）
        model_path: 模型路径
        output_dir: 输出目录
        port: 训练器端口号
    """
    # 1. 加载配置
    config = get_training_config(dataset_path, model_path, output_dir)
    
    # 2. 创建 Agent
    agent = LiteAgent(
        rollout_output_dir="./rollouts_1"
    )
    
    # 3. 创建算法和训练器
    algorithm = agl.VERL(config)
    
    adapter = TracerTraceToTriplet(
        agent_match=None,  # None 表示匹配所有 agent 节点
        llm_call_match=r"openai\.chat\.completion",  # 匹配 OpenAI chat completion 调用
        _skip_empty_token_spans=True
    )
    trainer = agl.Trainer(
        n_runners=64,
        algorithm=algorithm,
        adapter=adapter,
        port=port
    )
    
    # 4. 加载数据集
    train_file = config["data"]["train_files"]
    val_file = config["data"]["val_files"]
    
    # 检查文件是否存在
    if not Path(train_file).exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    if not Path(val_file).exists():
        raise FileNotFoundError(f"验证数据文件不存在: {val_file}")
    
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    
    # 检查数据是否为空
    if len(train_df) == 0:
        raise ValueError(f"训练数据文件为空: {train_file}")
    if len(val_df) == 0:
        raise ValueError(f"验证数据文件为空: {val_file}")
    
    # 检查必需的列是否存在
    required_columns = ["prompt", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"训练数据缺少必需的列: {missing_columns}")
    
    train_data = train_df.to_dict(orient="records")
    val_data = val_df.to_dict(orient="records")
    
    # 5. 开始训练
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RL Agent Training Script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/SFT_RL_bank/grok_merge.parquet",
        help="Path to the training dataset (parquet file)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/merge/sft_qwen3_0.6B",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/merge/sft_rl_qwen3_0.6B_new_reward",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=46549,
        help="Port number for the trainer (default: 32519)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 验证路径是否存在
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"数据集文件不存在: {args.dataset_path}")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
    
    # 创建输出目录（如果不存在）
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_fine_grained(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        port=args.port
    )

