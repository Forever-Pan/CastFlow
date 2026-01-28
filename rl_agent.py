import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import agentlightning as agl
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


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
                return {}
        
        # 如果解析结果是字典，直接返回
        if isinstance(parsed, dict):
            return parsed
        
        # 其他类型，返回空字典
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
        
    except (ValueError, TypeError):
        return {}


def calculate_mse(pred_dict: Dict[str, float], gt_dict: Dict[str, float]) -> float:
    """
    计算预测值和真实值之间的MSE
    
    Args:
        pred_dict: 预测值字典（时间戳 -> 数值）
        gt_dict: 真实值字典（时间戳 -> 数值）
        
    Returns:
        MSE值，如果无法计算则返回NaN
    """
    if not isinstance(pred_dict, dict) or not isinstance(gt_dict, dict):
        return float('nan')
    
    if not pred_dict or not gt_dict:
        return float('nan')
    
    # 按照时间戳对齐数据
    common_timestamps = set(pred_dict.keys()) & set(gt_dict.keys())
    
    if not common_timestamps:
        return float('nan')
    
    # 提取对齐后的数值
    pred_values = []
    gt_values = []
    
    # 按照时间戳排序
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
        return float('nan')
    
    # 计算MSE
    pred_array = np.array(pred_values)
    gt_array = np.array(gt_values)
    
    mse = np.mean((pred_array - gt_array) ** 2)
    
    return float(mse)


def build_small_model_prediction_dict(
    tool_str: str,
    gt_dict: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """
    从tool字段中解析小模型的reference_prediction，并根据ground_truth的时间戳对齐成字典
    
    Args:
        tool_str: 数据集中保存的tool列（JSON字符串）
        gt_dict: ground_truth解析后的字典，用于提供时间戳顺序
    
    Returns:
        小模型预测字典（时间戳 -> 数值），如果解析失败则返回None
    """
    if not tool_str or not gt_dict:
        return None
    
    try:
        tool_data = json.loads(tool_str)
    except Exception:
        return None
    
    # tool_data 预期是一个列表，包含若干tool调用结果
    if not isinstance(tool_data, list):
        return None
    
    reference_prediction = None
    for item in tool_data:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") == "model_auxiliary_tool":
            output = item.get("output") or {}
            reference_prediction = output.get("reference_prediction")
            break
    
    if not isinstance(reference_prediction, list):
        return None
    
    # 用ground_truth的时间戳顺序来对齐小模型的预测
    timestamps = sorted(gt_dict.keys())
    if len(timestamps) != len(reference_prediction):
        return None
    
    try:
        small_model_dict: Dict[str, float] = {}
        for ts, val in zip(timestamps, reference_prediction):
            small_model_dict[str(ts)] = float(val)
        return small_model_dict
    except (ValueError, TypeError):
        return None


def extract_answer_from_tags(answer_text: str) -> str:
    """
    从文本中提取 <answer></answer> 标签中的内容
    
    Args:
        answer_text: 可能包含 <answer></answer> 标签的文本
        
    Returns:
        提取出的答案内容，如果没有标签则返回原文本
    """
    if not answer_text:
        return ""
    
    # 尝试匹配 <answer>...</answer> 标签
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, answer_text, re.DOTALL)
    
    if matches:
        # 返回最后一个匹配的内容（通常只有一个）
        return matches[-1].strip()
    
    # 如果没有找到标签，返回原文本
    return answer_text.strip()
def compute_reward(answer: str, ground_truth: str, dataset_name: str) -> float:
    """
    计算reward，基于JSON解析有效性、长度匹配、时间戳匹配和MSE的分层判断
    
    Args:
        answer: 模型生成的答案（JSON格式字符串，可能包含<answer></answer>标签）
        ground_truth: 真实值（JSON格式字符串）
        
    Returns:
        reward值（浮点数）
        规则:
        1. JSON解析无效: -1
        2. 长度超出ground_truth: -0.5 - 0.3*x/24 (x为超出长度，惩罚至多为-0.8)
        3. 长度少于ground_truth: -0.8
        4. 长度一致但时间戳不匹配: -0.3
        5. 长度和时间戳都匹配: 基于MSE的奖励公式
    """
    # 从answer中提取<answer></answer>标签中的内容（如果有的话）
    answer_content = extract_answer_from_tags(answer)
    
    # 解析answer和ground_truth为字典
    answer_dict = parse_json_column(answer_content)
    gt_dict = parse_json_column(ground_truth)
    
    # 步骤1: 检查JSON解析是否有效
    # 如果原始内容不为空但解析后为空字典，说明解析失败
    if answer_content and answer_content.strip() and not answer_dict:
        print(f"[DEBUG] JSON解析无效 - answer_content不为空但解析后为空字典")
        return -1.0
    
    # 如果ground_truth解析失败，无法进行后续比较，返回-1
    if not gt_dict:
        print(f"[DEBUG] ground_truth解析失败")
        return -1.0
    
    # 步骤2: 判断长度是否与ground_truth一致
    answer_len = len(answer_dict)
    gt_len = len(gt_dict)
    
    if answer_len > gt_len:
        # 长度超出ground_truth: 惩罚 = -0.5 - 0.3*x/24，其中x为超出长度
        x = answer_len - gt_len
        penalty = -0.5 - 0.3 * x / (0.5*gt_len)
        # 惩罚至多为-0.8
        penalty = max(penalty, -0.8)
        print(f"[DEBUG] 长度超出ground_truth: answer_len={answer_len}, gt_len={gt_len}, 超出={x}, 惩罚={penalty:.6f}")
        return penalty
    elif answer_len < gt_len:
        # 长度少于ground_truth: 惩罚-0.8
        print(f"[DEBUG] 长度少于ground_truth: answer_len={answer_len}, gt_len={gt_len}, 惩罚=-0.8")
        return -0.8
    
    # 步骤3: 长度一致，判断时间戳是否与ground_truth一一匹配且一致
    answer_timestamps = set(answer_dict.keys())
    gt_timestamps = set(gt_dict.keys())
    
    if answer_timestamps != gt_timestamps:
        # 时间戳不匹配: 惩罚-0.3
        print(f"[DEBUG] 时间戳不匹配: answer_timestamps={sorted(answer_timestamps)[:5]}..., gt_timestamps={sorted(gt_timestamps)[:5]}...")
        return -0.3
    
    # 步骤4: 长度和时间戳都匹配，使用基于MSE的奖励公式
    mse = calculate_mse(answer_dict, gt_dict)
    
    # 如果MSE无效（理论上不应该发生，因为时间戳已匹配），返回-1
    if np.isnan(mse):
        print(f"[DEBUG] MSE is NaN - 时间戳已匹配但MSE计算失败")
        return -1.0
    
    # 将MSE转换为reward的分段函数:
    # - 当 MSE ∈ [0, 5000] 时: reward = 1 - 0.9 * sin(pi * mse / 10000)
    # - 当 MSE > 5000 时: reward = 0.3 * exp(-ln(3) * mse / 5000)
    upper_bound = 10
    print("dataset_name:",dataset_name)
    if dataset_name == "NP":
        upper_bound = 40
    elif dataset_name == "BE":
        upper_bound = 800
    elif dataset_name == "DE":
        upper_bound = 350
    elif dataset_name == "FR":
        upper_bound = 1000
    elif dataset_name == "PJM":
        upper_bound = 45
    elif dataset_name == "MOPEX":
        upper_bound = 7
    elif dataset_name == "sunny":
        upper_bound = 50
    elif dataset_name == "windy":
        upper_bound = 3000
    elif dataset_name =="ETTh1":
        upper_bound = 12
    elif dataset_name == "ETTm1":
        upper_bound = 3.5

    if mse <= upper_bound:
        reward = 1 - 0.9 * np.sin(np.pi * mse / (2*upper_bound))
    else:
        reward = 0.3 * np.exp((-np.log(3) * mse) / upper_bound)
    
    # 调试信息：打印MSE和reward
    print(f"[DEBUG] MSE: {mse:.6f}, reward: {reward:.6f}, answer_dict_size: {len(answer_dict)}, gt_dict_size: {len(gt_dict)}")
    
    return reward


def compute_contrastive_reward(
    answer: str,
    ground_truth: str,
    dataset_name: str,
    tool: Optional[str] = None,
) -> float:
    """
    对比学习版reward，由两部分组成：
    1. 绝对MSE部分：直接复用原始的 compute_reward（answer vs ground_truth），记为 base_reward，占主奖励；
    2. 对比部分：使用 (mse_small - mse_ans) / (upper_bound / 500) 作为额外加成/惩罚：
       - mse_ans: 当前answer相对ground_truth的MSE
       - mse_small: 小模型(reference_prediction)相对ground_truth的MSE
       - 当 mse_ans < mse_small（大模型比小模型好）时，这一项为正；反之为负
    其他格式检查逻辑与原compute_reward保持一致（JSON有效性、长度一致、时间戳匹配）。
    """
    # 从answer中提取<answer></answer>标签中的内容（如果有的话）
    answer_content = extract_answer_from_tags(answer)
    
    # 解析answer和ground_truth为字典
    answer_dict = parse_json_column(answer_content)
    gt_dict = parse_json_column(ground_truth)
    
    # 步骤1: 检查JSON解析是否有效
    if answer_content and answer_content.strip() and not answer_dict:
        print(f"[DEBUG] JSON解析无效 - answer_content不为空但解析后为空字典")
        return -1.0
    
    if not gt_dict:
        print(f"[DEBUG] ground_truth解析失败")
        return -1.0
    
    # 步骤2: 判断长度是否与ground_truth一致
    answer_len = len(answer_dict)
    gt_len = len(gt_dict)
    
    if answer_len > gt_len:
        x = answer_len - gt_len
        penalty = -0.5 - 0.3 * x / (0.5 * gt_len)
        penalty = max(penalty, -0.8)
        print(f"[DEBUG][contrastive] 长度超出ground_truth: answer_len={answer_len}, gt_len={gt_len}, 超出={x}, 惩罚={penalty:.6f}")
        return penalty
    elif answer_len < gt_len:
        print(f"[DEBUG][contrastive] 长度少于ground_truth: answer_len={answer_len}, gt_len={gt_len}, 惩罚=-0.8")
        return -0.8
    
    # 步骤3: 长度一致，判断时间戳是否与ground_truth一一匹配且一致
    answer_timestamps = set(answer_dict.keys())
    gt_timestamps = set(gt_dict.keys())
    
    if answer_timestamps != gt_timestamps:
        print(
            f"[DEBUG][contrastive] 时间戳不匹配: "
            f"answer_timestamps={sorted(answer_timestamps)[:5]}..., "
            f"gt_timestamps={sorted(gt_timestamps)[:5]}..."
        )
        return -0.3
    
    # 步骤4: 计算answer相对于ground_truth的MSE
    mse_ans = calculate_mse(answer_dict, gt_dict)
    if np.isnan(mse_ans):
        print(f"[DEBUG][contrastive] mse_ans is NaN")
        return -1.0
    
    # 先计算绝对MSE部分的reward（复用原compute_reward）
    base_reward = compute_reward(answer, ground_truth, dataset_name)
    
    # 如果没有tool或解析不到小模型预测，则只使用绝对MSE reward
    if not tool:
        print(f"[DEBUG][contrastive] 无tool信息，仅使用base_reward: {base_reward:.6f}")
        return base_reward
    
    small_model_dict = build_small_model_prediction_dict(tool, gt_dict)
    if not small_model_dict:
        print("[DEBUG][contrastive] 无法从tool中解析小模型reference_prediction，仅使用base_reward")
        return base_reward
    
    mse_small = calculate_mse(small_model_dict, gt_dict)
    if np.isnan(mse_small):
        print(f"[DEBUG][contrastive] mse_small is NaN，仅使用base_reward")
        return base_reward
    print("dataset_name:",dataset_name)
    # 根据dataset_name选择upper_bound，用于对比项的缩放
    upper_bound = 10
    if dataset_name == "NP":
        upper_bound = 40
    elif dataset_name == "BE":
        upper_bound = 800
    elif dataset_name == "DE":
        upper_bound = 350
    elif dataset_name == "FR":
        upper_bound = 1000
    elif dataset_name == "PJM":
        upper_bound = 45
    elif dataset_name == "MOPEX":
        upper_bound = 7
    elif dataset_name == "sunny":
        upper_bound = 50
    elif dataset_name == "windy":
        upper_bound = 3000
    elif dataset_name == "ETTh1":
        upper_bound = 12
    elif dataset_name == "ETTm1":
        upper_bound = 3.5
    
    # 对比项： (mse_small - mse_ans) / (upper_bound / 500)
    # 按你的要求，这一项最终应被bound到 [-0.5, 0.5]
    denom = upper_bound if upper_bound > 0 else 1.0
    contrast_raw = (mse_small - mse_ans) / denom * 100
    contrast_term = max(min(contrast_raw, 0.5), -0.5)
    
    # 总reward = 绝对MSE reward + 对比项
    reward = base_reward + contrast_term
    
    print(
        f"[DEBUG][contrastive] mse_ans: {mse_ans:.6f}, mse_small: {mse_small:.6f}, "
        f"base_reward: {base_reward:.6f}, contrast_term: {contrast_term:.6f}, final_reward: {reward:.6f}"
    )
    
    return reward


class LiteAgent(agl.LitAgent[Dict[str, Any]]):
    _task_counter = 0

    def __init__(
        self,
        rollout_output_dir: Optional[str] = "./rollouts",
    ) -> None:
        super().__init__()
        self.rollout_output_dir = Path(rollout_output_dir)
        self.rollout_output_dir.mkdir(parents=True, exist_ok=True)
        # 映射prompt的hash到task索引，确保相同prompt使用相同索引
        self._prompt_to_idx: Dict[int, int] = {}
    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        """执行 rollout，并记录所有 Agent 的输入输出
        
        使用 langsmith traceable 装饰器进行监控，并保存完整的执行记录到文件夹。
        """
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        attempted_rollout = cast(agl.AttemptedRollout, rollout)
        
        # 获取正确的端点URL（包含rollout和attempt信息）
        base_url = llm.get_base_url(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id)
        
        # 创建LangChain聊天模型
        chat_model = init_chat_model(
            model="hosted_vllm/" + llm.model,
            model_provider="openai",
            openai_api_base=base_url,
            openai_api_key=llm.api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=llm.sampling_parameters.get("temperature", 0.0),
        )
        
        prompt = task["prompt"]  
        dataset_name = task["dataset_name"]
        messages = [HumanMessage(content=prompt)]
        
        # 关键修复：使用 tracer 的 langchain handler 来记录 LLM 调用
        # 这样 adapter 才能从 trace 中提取 triplets
        handler = self.tracer.get_langchain_handler()
        if handler:
            # LangChain ChatModel 支持通过 with_config 或直接传递 config
            # 使用 RunnableConfig 格式
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(callbacks=[handler])
            result = chat_model.invoke(messages, config=config)
        else:
            result = chat_model.invoke(messages)
        answer = result.content if hasattr(result, 'content') else str(result)
        ground_truth = task.get("ground_truth", "")
        tool = task.get("tool", "")
        
        # 计算MSE和reward
        answer_content = extract_answer_from_tags(answer)
        answer_dict = parse_json_column(answer_content)
        gt_dict = parse_json_column(ground_truth)
        mse = calculate_mse(answer_dict, gt_dict)
        # 使用对比学习版reward：与小模型reference_prediction进行对比
        reward = compute_contrastive_reward(answer, ground_truth, dataset_name, tool=tool)
        
        # 确保 reward 始终是有效的 float
        if reward is None or np.isnan(reward) or np.isinf(reward):
            reward = -1.0
        
        # 获取task索引：优先使用task中的idx，否则使用prompt的稳定hash
        task_idx = self._get_task_idx(task, prompt)
        
        # 保存rollout信息到文件
        self._save_rollout(
            rollout_id=attempted_rollout.rollout_id,
            prompt=prompt,
            answer=answer,
            ground_truth=ground_truth,
            reward=float(reward),
            mse=float(mse) if not np.isnan(mse) else None,
            task_idx=task_idx
        )
        
        print(f"rollout reward: {reward}")
        return float(reward)
    
    def _get_task_idx(self, task: Dict[str, Any], prompt: str) -> int:
        """
        获取task索引：优先使用task中的idx，否则使用prompt的MD5哈希
        
        Args:
            task: 任务字典，可能包含idx字段
            prompt: prompt字符串
            
        Returns:
            task索引（从1开始）
        """
        # 优先使用task中的idx字段（如果存在且有效）
        if "idx" in task and task["idx"] is not None:
            try:
                idx = int(task["idx"])
                # idx通常是0-based，转为1-based用于目录命名
                return idx + 1
            except (ValueError, TypeError):
                pass
        
        # 如果没有idx，使用prompt的MD5哈希的前8位作为稳定标识符
        # MD5在不同进程间是稳定的，比Python的hash()更可靠
        prompt_md5 = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
        prompt_hash_key = int(prompt_md5, 16)  # 将16进制字符串转为整数
        
        if prompt_hash_key not in self._prompt_to_idx:
            LiteAgent._task_counter += 1
            self._prompt_to_idx[prompt_hash_key] = LiteAgent._task_counter
        return self._prompt_to_idx[prompt_hash_key]
    
    def _save_rollout(
        self,
        rollout_id: str,
        prompt: str,
        answer: str,
        ground_truth: str,
        reward: float,
        mse: Optional[float],
        task_idx: int,
    ) -> None:
        """保存rollout信息到JSON文件"""
        # 创建prompt目录: rollouts/prompt_0/
        prompt_dir = self.rollout_output_dir / f"prompt_{task_idx}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名: rollout_id.json (去掉 "ro-" 前缀以简化文件名)
        filename = rollout_id.replace("ro-", "") + ".json"
        filepath = prompt_dir / filename
        
        # 保存信息（确保使用高精度）
        # 将numpy类型转换为Python原生类型，确保JSON序列化时保持完整精度
        rollout_data = {
            "rollout_id": rollout_id,
            "prompt": prompt,
            "answer": answer,
            "ground_truth": ground_truth,
            # 转换为Python原生float，json.dump会自动保持约15-17位有效数字的精度
            "reward": float(reward) if not (np.isnan(reward) or np.isinf(reward)) else None,
            "mse": float(mse) if (mse is not None and not np.isnan(mse) and not np.isinf(mse)) else None,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            # Python的json.dump默认保持双精度浮点数精度（约15-17位有效数字）
            # 这对于大多数应用已经足够精确
            json.dump(rollout_data, f, ensure_ascii=False, indent=2)

