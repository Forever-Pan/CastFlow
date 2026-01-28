#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 PEFT/LoRA 适配器到基础模型
用于生成完整的模型文件，以便 vLLM 可以加载
"""
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """
    合并适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: 适配器路径
        output_path: 合并后的模型输出路径
    """
    print(f"📥 加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"📥 加载适配器: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("🔄 合并适配器到基础模型...")
    merged_model = model.merge_and_unload()
    
    print(f"💾 保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    # 保存 tokenizer
    print("💾 保存 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ 合并完成！合并后的模型保存在: {output_path}")
    print(f"   现在可以使用 vLLM 加载: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 PEFT 适配器到基础模型")
    parser.add_argument(
        "--base-model",
        type=str,
        default="./models/Qwen3-0.6B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./models/sft_model",
        help="适配器路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/sft_model_merged",
        help="合并后的模型输出路径"
    )
    
    args = parser.parse_args()
    
    merge_adapter(args.base_model, args.adapter, args.output)

