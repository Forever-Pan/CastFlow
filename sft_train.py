#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT训练脚本
基于CSV数据集进行监督式微调训练，支持Qwen3-4B等模型

使用方法:
    # 基本使用（使用默认参数）
    python sft_train.py
    
    # 指定数据集、模型和输出路径
    python sft_train.py \
        --dataset_path ./datasets/SFT_RL_bank/ETTH1/grok_final.csv \
        --model_path ./models/Qwen3-8B \
        --output_dir ./models/ETTH1/sft_Qwen3_8B
    
    # 多GPU训练
    torchrun --nproc_per_node=1 --master_port=32588 sft_train.py \
     --model_path ./models/Qwen3-0.6B \
     --dataset_path ./datasets/SFT_RL_bank/sunny/grok.csv \
     --output_dir ./models/sunny/sft_qwen3_4b
"""

import os
import sys
import argparse
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

# 尝试导入PEFT（用于LoRA）
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  警告: 未安装peft库，LoRA功能将不可用。安装命令: pip install peft")
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
# ============ 1. 解析命令行参数 ============ 
parser = argparse.ArgumentParser(
    description='SFT训练脚本 - 支持Qwen3-4B等模型的监督式微调',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# 核心参数：数据集、模型路径、输出目录
parser.add_argument(
    '--dataset_path', 
    type=str, 
    default='./datasets/SFT_RL_bank/grok_merge.csv',
    help='训练数据集CSV文件路径（必须包含prompt和response列）'
)
parser.add_argument(
    '--model_path', 
    type=str, 
    default='./models/Qwen3-4B',
    help='本地LLM模型路径（支持Qwen3-4B等模型）'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    default='./models/merge/sft_qwen3_0.6B',
    help='训练后模型的保存目录'
)

# 训练超参数
parser.add_argument('--batch_size', type=int, default=1, help='每设备batch size（Qwen3-4B建议：A800=2-4，4090=1-2）')
parser.add_argument('--gradient_accumulation', type=int, default=8, help='梯度累积步数（Qwen3-4B建议：A800=2-4，4090=4-8）')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率（LoRA建议1e-4，全量微调建议5e-5）')
parser.add_argument('--num_epochs', type=int, default=2, help='训练轮数（建议3-5轮）')
parser.add_argument('--max_length', type=int, default=14000, help='最大序列长度（Qwen3-4B支持8192）')
parser.add_argument('--save_strategy', type=str, default='steps', choices=['no', 'steps', 'epoch'], help='模型保存策略')
parser.add_argument('--save_steps', type=int, default=275, help='按steps保存时的步数（仅当save_strategy=steps时生效）')
parser.add_argument('--save_total_limit', type=int, default=5, help='最多保留的checkpoint数量')
parser.add_argument('--use_deepspeed', action='store_true', help='是否使用DeepSpeed（默认开启，适合大模型）')
parser.add_argument('--no_deepspeed', dest='use_deepspeed', action='store_false', help='禁用DeepSpeed')
parser.set_defaults(use_deepspeed=True)  # 设置默认值为True
parser.add_argument('--use_lora', action='store_true', help='是否使用LoRA（默认关闭，可显式指定开启）')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank（默认16，可调8/32/64）')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha（默认32，通常为rank的2倍）')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout（默认0.05）')
parser.add_argument('--lora_target_modules', type=str, default="", help='LoRA目标模块，逗号分隔（默认自动检测Qwen模型）')

args = parser.parse_args()

# 核心参数
DATASET_PATH = args.dataset_path
MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

# 验证核心参数
if not os.path.exists(DATASET_PATH):
    print(f"❌ 错误: 数据集文件不存在: {DATASET_PATH}")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 模型路径不存在: {MODEL_PATH}")
    print(f"   请确保模型已下载到指定路径，或使用正确的模型路径")
    sys.exit(1)

print("=" * 80)
print("SFT训练配置")
print("=" * 80)
print(f"  数据集路径: {DATASET_PATH}")
print(f"  模型路径: {MODEL_PATH}")
print(f"  输出目录: {OUTPUT_DIR}")
print("=" * 80)
# 训练超参数
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.gradient_accumulation
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
MAX_LENGTH = args.max_length
SAVE_STRATEGY = args.save_strategy
SAVE_STEPS = args.save_steps
SAVE_TOTAL_LIMIT = args.save_total_limit
USE_DEEPSPEED = args.use_deepspeed
USE_LORA = args.use_lora
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
LORA_TARGET_MODULES = args.lora_target_modules.split(',') if args.lora_target_modules else None

# ============ 2. DeepSpeed 配置（可选） ============ 
ds_config = None
if USE_DEEPSPEED:
    try:
        import deepspeed
        ds_config = {
            "fp16": {"enabled": False},
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,                      # ZeRO-2 适合 8B 模型在 4x80G 跑全量
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 10,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False
        }
        print("✅ 已启用DeepSpeed优化")
    except ImportError:
        print("⚠️  警告: 未安装deepspeed，将不使用DeepSpeed优化")
        USE_DEEPSPEED = False
        ds_config = None
else:
    print("ℹ️  未启用DeepSpeed，使用标准训练模式")

# ============ 3. 加载分词器 ============ 
print(f"\n正在加载分词器: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"✅ 分词器加载完成，词汇表大小: {len(tokenizer)}")

# ============ 4. 从CSV加载数据并转换为messages格式 ============ 
print(f"\n正在加载数据集: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, keep_default_na=False)
print(f"  原始数据行数: {len(df)}")

# 检查必需的列
required_columns = ['prompt', 'response']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"❌ 错误: 数据集缺少必需的列: {missing_columns}")
    print(f"   数据集包含的列: {list(df.columns)}")
    sys.exit(1)

# 过滤掉response为空的行
df = df[df['response'].str.strip() != '']
print(f"  有效数据行数: {len(df)}")

# 转换为messages格式
def csv_to_messages(row):
    """将CSV行转换为chat messages格式"""
    prompt = str(row['prompt']).strip()
    response = str(row['response']).strip()
    
    # 确保prompt和response都不为空
    if not prompt or not response:
        return None
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"messages": messages}

# 转换为Dataset
data_list = []
for idx, row in df.iterrows():
    try:
        messages = csv_to_messages(row)
        if messages is not None:
            data_list.append(messages)
    except Exception as e:
        if idx < 10:  # 只显示前10个错误的详细信息
            print(f"  警告: 跳过第{idx}行数据，错误: {str(e)}")
        continue

if len(data_list) == 0:
    print("❌ 错误: 没有有效的数据可以训练")
    sys.exit(1)

print(f"✅ 成功转换 {len(data_list)} 条数据")
dataset = Dataset.from_list(data_list)

# ============ 5. 分词与 Mask 逻辑 ============ 
def tokenize_multiturn_chat(example):
    """对多轮对话进行分词和标签mask"""
    messages = example["messages"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding=False, add_special_tokens=False)
    
    input_ids = list(full_tokenized["input_ids"])
    labels = list(input_ids)
    
    # 只对assistant的回复部分计算loss，其他部分mask掉
    prev_len = 0
    for i in range(1, len(messages) + 1):
        curr_all_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
        curr_tokenized = tokenizer(curr_all_text, add_special_tokens=False)
        curr_len = len(curr_tokenized["input_ids"])
        
        is_assistant = (messages[i-1]["role"] == "assistant")
        is_last_message = (i == len(messages))
        
        # 只保留最后一个assistant消息的loss，其他都mask
        if not (is_assistant and is_last_message):
            start_idx = prev_len
            end_idx = min(curr_len, len(labels))
            if start_idx < end_idx:
                for idx in range(start_idx, end_idx):
                    labels[idx] = -100
        prev_len = curr_len
        if prev_len >= len(labels):
            break
                
    return {"input_ids": input_ids, "attention_mask": full_tokenized["attention_mask"], "labels": labels}

print("\n正在对数据进行分词...")
tokenized_dataset = dataset.map(
    tokenize_multiturn_chat, 
    remove_columns=dataset.column_names, 
    num_proc=min(8, os.cpu_count())
)
print(f"✅ 分词完成，数据集大小: {len(tokenized_dataset)}")

# ============ 6. 加载模型并应用LoRA（如果启用） ============ 
print(f"\n正在加载模型: {MODEL_PATH}")
print("  这可能需要一些时间，请耐心等待...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"✅ 模型加载完成")

# 应用LoRA配置
if USE_LORA:
    if not HAS_PEFT:
        print("❌ 错误: 未安装peft库，无法使用LoRA。请运行: pip install peft")
        sys.exit(1)
    
    # 自动检测目标模块（如果未指定）
    if LORA_TARGET_MODULES is None:
        # 根据模型类型自动选择目标模块
        model_type = getattr(model.config, 'model_type', '').lower()
        model_name_lower = MODEL_PATH.lower()
        
        if 'qwen' in model_type or 'qwen' in model_name_lower:
            # Qwen3模型使用这些模块
            LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(f"  检测到Qwen模型，使用Qwen专用LoRA目标模块")
        elif 'llama' in model_type or 'llama' in model_name_lower:
            LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
            print(f"  检测到LLaMA模型，使用LLaMA专用LoRA目标模块")
        else:
            # 通用配置，尝试常见的attention模块
            LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
            print(f"⚠️  警告: 未识别的模型类型 {model_type}，使用默认LoRA目标模块")
    
    print(f"✅ 启用LoRA配置:")
    print(f"   Rank (r): {LORA_R}")
    print(f"   Alpha: {LORA_ALPHA}")
    print(f"   Dropout: {LORA_DROPOUT}")
    print(f"   目标模块: {LORA_TARGET_MODULES}")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    print("ℹ️  未启用LoRA，使用全量微调")

# ============ 7. Data Collator ============ 
@dataclass
class ToolDataCollator:
    tokenizer: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# ============ 8. 训练配置 ============ 
training_args_dict = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": NUM_EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "logging_steps": 1,
    "save_strategy": SAVE_STRATEGY,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": False
}

# 只在启用DeepSpeed时添加deepspeed配置
if USE_DEEPSPEED and ds_config is not None:
    training_args_dict["deepspeed"] = ds_config

training_args = TrainingArguments(**training_args_dict)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=ToolDataCollator(tokenizer)
)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 80)
print("🚀 启动SFT训练")
print("=" * 80)
print(f"  模型路径: {MODEL_PATH}")
print(f"  数据集: {DATASET_PATH} ({len(tokenized_dataset)} 条样本)")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  训练配置:")
print(f"    - Batch Size: {BATCH_SIZE}")
print(f"    - Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print(f"    - Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION * torch.cuda.device_count() if torch.cuda.is_available() else BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"    - Learning Rate: {LEARNING_RATE}")
print(f"    - Epochs: {NUM_EPOCHS}")
print(f"    - Max Length: {MAX_LENGTH}")
print(f"    - DeepSpeed: {'✅ 启用' if USE_DEEPSPEED and ds_config else '❌ 未启用'}")
print(f"    - LoRA: {'✅ 启用' if USE_LORA else '❌ 未启用'}")
if USE_LORA:
    print(f"      * Rank: {LORA_R}, Alpha: {LORA_ALPHA}, Dropout: {LORA_DROPOUT}")
    if LORA_TARGET_MODULES:
        print(f"      * 目标模块: {', '.join(LORA_TARGET_MODULES)}")
print("=" * 80)
print()

trainer.train()

# ============ 9. 保存模型 ============ 
if trainer.is_world_process_zero() or not torch.distributed.is_initialized():
    print("\n" + "=" * 80)
    print("💾 保存训练后的模型")
    print("=" * 80)
    
    if USE_LORA:
        # LoRA模式：只保存adapter权重
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ SFT训练完成！")
        print(f"   LoRA adapter已保存至: {OUTPUT_DIR}")
        print(f"   ⚠️  注意: 这是LoRA adapter权重，使用时需要:")
        print(f"      1. 加载基础模型: {MODEL_PATH}")
        print(f"      2. 加载adapter: {OUTPUT_DIR}")
    else:
        # 全量微调：保存完整权重
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ SFT训练完成！")
        print(f"   完整模型已保存至: {OUTPUT_DIR}")
        print(f"   可以直接使用该路径加载模型")
    
    print("=" * 80)

