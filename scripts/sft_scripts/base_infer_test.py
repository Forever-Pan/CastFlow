#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate answers on the test split using the base local LLM (pre-SFT) served via vLLM.
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载 .env 配置
try:
    from castmaster.env import load_env, get_openai_config
    load_env()
except ImportError:
    # 如果导入失败，尝试使用 dotenv
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            project_root = os.path.dirname(os.path.abspath(__file__))
            env_file = os.path.join(project_root, '..', '..', '.env')
            if os.path.exists(env_file):
                load_dotenv(env_file, override=False)
    except ImportError:
        pass  # 如果没有dotenv，使用系统环境变量

from call_api_parallel import process_csv_parallel


def main():
    parser = argparse.ArgumentParser(description="Base model inference on test split")
    parser.add_argument(
        "--input",
        default="./datasets/windy_power_processed_test.csv",
        help="Input prompt CSV",
    )
    parser.add_argument(
        "--output",
        default="./datasets/base_test_answers.csv",
        help="Output CSV for base model responses",
    )
    parser.add_argument("--workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--timeout", type=float, default=240.0, help="Request timeout seconds")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (overrides .env)")
    parser.add_argument("--model", default=None, help="Served model name (overrides .env)")
    parser.add_argument("--api-key", default=None, help="API key (overrides .env)")
    args = parser.parse_args()

    # 从 .env 加载配置
    try:
        config = get_openai_config()
    except NameError:
        # 如果 get_openai_config 不存在，从环境变量读取
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("MODEL"),
        }

    # 命令行参数优先（覆盖 .env 配置）
    if args.base_url:
        config["base_url"] = args.base_url
        os.environ["OPENAI_BASE_URL"] = args.base_url
    elif config.get("base_url"):
        os.environ["OPENAI_BASE_URL"] = config["base_url"]
    
    if args.model:
        config["model"] = args.model
        os.environ["MODEL"] = args.model
    elif config.get("model"):
        os.environ["MODEL"] = config["model"]
    
    if args.api_key:
        config["api_key"] = args.api_key
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif config.get("api_key"):
        os.environ["OPENAI_API_KEY"] = config["api_key"]

    # 显示使用的配置（不显示完整密钥）
    print("配置信息（从 .env 加载，命令行参数可覆盖）:")
    print(f"  Model: {config.get('model', '未设置')}")
    print(f"  Base URL: {config.get('base_url', '未设置')}")
    if config.get("api_key"):
        api_key_preview = config['api_key'][:8] + "..." if len(config['api_key']) > 8 else "***"
        print(f"  API Key: {api_key_preview} (已设置)")
    else:
        print(f"  API Key: 未设置")
    print("-" * 50)

    process_csv_parallel(args.input, args.output, max_workers=args.workers, timeout=args.timeout)


if __name__ == "__main__":
    main()
