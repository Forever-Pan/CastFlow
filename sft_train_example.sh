#!/bin/bash
# SFT训练脚本使用示例

# 示例1: 使用默认参数训练（使用默认的数据集和模型路径）
# python sft_train.py

# 示例2: 指定数据集、模型路径和输出目录
python sft_train.py \
    --dataset_path ./datasets/SFT_RL_bank/windy/grok.csv \
    --model_path ./models/Qwen3-4B \
    --output_dir ./models/sft_qwen3_4b_windy

# 示例3: 使用LoRA进行参数高效微调（推荐，节省显存）
python sft_train.py \
    --dataset_path ./datasets/SFT_RL_bank/windy/grok.csv \
    --model_path ./models/Qwen3-4B \
    --output_dir ./models/sft_qwen3_4b_windy_lora \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --learning_rate 1e-4 \
    --num_epochs 3

# 示例4: 全量微调（需要更多显存）
python sft_train.py \
    --dataset_path ./datasets/SFT_RL_bank/windy/grok.csv \
    --model_path ./models/Qwen3-4B \
    --output_dir ./models/sft_qwen3_4b_windy_full \
    --no_lora \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --num_epochs 3

# 示例5: 多GPU训练（使用torchrun）
# torchrun --nproc_per_node=4 --master_port=40888 sft_train.py \
#     --dataset_path ./datasets/SFT_RL_bank/windy/grok.csv \
#     --model_path ./models/Qwen3-4B \
#     --output_dir ./models/sft_qwen3_4b_windy \
#     --batch_size 2 \
#     --gradient_accumulation 4
