#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补丁脚本：将 verl 的 ZeroMQ 锁和 IPC 路径从 /tmp 改为 ~/.ray_tmp，
解决 /tmp/verl_vllm_zmq.lock 权限不足 (PermissionError) 问题。

用法（使用与 rl_agent_train 相同的 conda 环境，例如 CastMaster_new）：
    python scripts/rl_scripts/patch_verl_zmq_path.py
  或： /path/to/anaconda3/envs/CastMaster_new/bin/python scripts/rl_scripts/patch_verl_zmq_path.py

升级 verl 或 agentlightning 后若问题复现，请重新运行此脚本。
"""

import os
import sys
from pathlib import Path

def find_vllm_rollout_spmd():
    try:
        import verl.workers.rollout.vllm_rollout.vllm_rollout_spmd as m
        return Path(m.__file__).resolve()
    except Exception as e:
        print(f"无法定位 verl 模块: {e}")
        return None

def main():
    p = find_vllm_rollout_spmd()
    if not p or not p.exists():
        print("未找到 verl 的 vllm_rollout_spmd.py，请确认已安装 verl。")
        sys.exit(1)

    path = p
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 若已改为 FileLock(lock_path)，视为已打补丁
    if 'FileLock("/tmp/verl_vllm_zmq.lock")' not in content:
        print(f"该文件似已打过补丁或版本不同，跳过: {path}")
        return

    old = '''        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"'''

    new = '''        # Use VERL_ZMQ_DIR or ~/.ray_tmp to avoid /tmp permission issues.
        zmq_dir = os.environ.get("VERL_ZMQ_DIR", os.path.join(os.path.expanduser("~"), ".ray_tmp"))
        os.makedirs(zmq_dir, exist_ok=True)
        lock_path = os.path.join(zmq_dir, "verl_vllm_zmq.lock")

        # File lock to prevent multiple workers listen to same port
        with FileLock(lock_path):
            if socket_type == "ipc":
                pid = os.getpid()
                ipc_path = os.path.join(zmq_dir, f"verl_vllm_zmq_{pid}.ipc")
                address = "ipc://" + os.path.abspath(ipc_path)'''

    if old not in content:
        print(f"未在文件中找到预期的代码块，可能版本不一致。路径: {path}")
        sys.exit(1)

    # 备份
    bak = path.with_suffix(path.suffix + ".bak")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已备份: {bak}")

    new_content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"补丁已应用: {path}")
    print("可重新运行 rl_agent_train.py。若升级 verl/agentlightning 后问题复现，请再次执行本脚本。")

if __name__ == "__main__":
    main()
