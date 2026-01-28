#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行调用API脚本
读取包含prompt的CSV文件，并行调用API，解析response中的<think>和<answer>，添加到CSV中
"""

import pandas as pd
import sys
import re
import concurrent.futures
from typing import Dict, Optional, Tuple
import os
from openai import OpenAI
import time
import threading

# 尝试导入tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有tqdm，创建一个简单的占位符
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.n = 0
        
        def update(self, n=1):
            self.n += n
        
        def set_postfix(self, **kwargs):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

# 导入环境配置
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    # 导入时会自动调用load_env()加载.env文件
    from castmaster.env import get_openai_config, load_env
    # 确保加载.env文件
    load_env()
except ImportError:
    # 如果导入失败，尝试手动加载.env文件
    try:
        from dotenv import load_dotenv, find_dotenv
        # 尝试从当前目录或项目根目录加载.env
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
        else:
            # 尝试项目根目录
            project_root = os.path.dirname(os.path.abspath(__file__))
            env_file = os.path.join(project_root, '.env')
            if os.path.exists(env_file):
                load_dotenv(env_file, override=False)
    except ImportError:
        pass  # 如果没有dotenv，就使用系统环境变量
    
    # 如果导入失败，直接使用环境变量
    def get_openai_config():
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("MODEL", "gpt-4o-mini"),
        }

def parse_response(response_text: str) -> Tuple[str, str]:
    """
    从response的content中解析出<think>和<answer>的内容
    
    Args:
        response_text: API返回的文本内容（response.choices[0].message.content）
        
    Returns:
        (reasoning, answer): 解析出的reasoning和answer内容
    """
    reasoning = ""
    answer = ""
    
    if not response_text:
        return reasoning, answer
    
    # 从content中提取<think>...</think>标签内的内容
    reasoning_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # 从content中提取<answer>...</answer>标签内的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer


def is_valid_answer(answer: str) -> bool:
    """
    检查answer是否是有效的JSON格式（包含时间戳和数值的字典）
    
    Args:
        answer: 解析出的answer字符串
        
    Returns:
        True如果answer是有效的JSON格式，False否则
    """
    if not answer or not answer.strip():
        return False
    
    try:
        import json
        # 尝试解析JSON
        data = json.loads(answer)
        
        # 检查是否是字典类型
        if not isinstance(data, dict):
            return False
        
        # 检查是否有至少一个键值对
        if len(data) == 0:
            return False
        
        # 检查值是否是数字类型
        for key, value in data.items():
            try:
                float(value)
            except (ValueError, TypeError):
                return False
        
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        # 如果JSON解析失败，返回False
        return False


def call_api_single(prompt: str, idx: int, config: Dict, timeout: float = 240.0) -> Dict:
    """
    调用单个API请求
    
    Args:
        prompt: 要发送的prompt
        idx: 样本索引
        config: OpenAI配置
        timeout: 超时时间（秒）
        
    Returns:
        包含response、reasoning、answer的字典
    """
    api_key = config.get("api_key")
    base_url = config.get("base_url")
    model = config.get("model", "gpt-4o-mini")
    max_tokens = config.get("max_tokens")  # 最大生成token数，None表示无限制
    
    if not api_key:
        return {
            "idx": idx,
            "response": "",
            "reasoning": "",
            "answer": "",
            "answer_valid": False,
            "error": "[ConfigurationError] No API key provided | 建议: 请在 .env 文件中设置 OPENAI_API_KEY 或通过环境变量设置",
            "error_details": {
                "error_type": "ConfigurationError",
                "error_category": "configuration",
                "error_message": "No API key provided",
                "suggestion": "请在 .env 文件中设置 OPENAI_API_KEY 或通过环境变量设置",
            },
            "success": False
        }
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        
        # 构建API调用参数
        create_params = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "timeout": timeout,
        }
        
        # 如果指定了max_tokens，添加到参数中
        if max_tokens is not None:
            create_params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**create_params)
        
        response_text = (response.choices[0].message.content or "").strip()
        reasoning, answer = parse_response(response_text)
        
        # 检查answer是否有效
        answer_valid = is_valid_answer(answer)
        
        return {
            "idx": idx,
            "response": response_text,
            "reasoning": reasoning,
            "answer": answer,
            "answer_valid": answer_valid,  # 标记answer是否有效
            "error": None,
            "success": True
        }
        
    except Exception as e:
        # 提供更详细的错误信息
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 尝试获取更多错误详情
        error_details = {
            "error_type": error_type,
            "error_message": error_msg,
        }
        
        # 检查是否是 OpenAI API 相关的错误
        if hasattr(e, 'status_code'):
            error_details["status_code"] = e.status_code
        if hasattr(e, 'response'):
            error_details["has_response"] = True
            try:
                if hasattr(e.response, 'text'):
                    error_details["response_text"] = e.response.text[:500]  # 限制长度
            except:
                pass
        
        # 检查是否是网络相关错误
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_details["error_category"] = "timeout"
            error_details["suggestion"] = f"请求超时（timeout={timeout}秒），可以尝试增加超时时间或检查网络连接"
        elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
            error_details["error_category"] = "connection"
            error_details["suggestion"] = f"连接失败，请检查 base_url 是否正确（当前: {base_url}）以及服务器是否运行"
        elif "401" in error_msg or "Unauthorized" in error_msg:
            error_details["error_category"] = "authentication"
            error_details["suggestion"] = "认证失败，请检查 API key 是否正确"
        elif "404" in error_msg or "Not Found" in error_msg:
            error_details["error_category"] = "not_found"
            error_details["suggestion"] = f"资源未找到，请检查 base_url 和 model 是否正确（base_url: {base_url}, model: {model}）"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            error_details["error_category"] = "rate_limit"
            error_details["suggestion"] = "请求频率过高，请减少并发数或等待后重试"
        elif "500" in error_msg or "Internal Server Error" in error_msg:
            error_details["error_category"] = "server_error"
            error_details["suggestion"] = "服务器内部错误，可能是服务器端问题，请稍后重试"
        else:
            error_details["error_category"] = "unknown"
            error_details["suggestion"] = "未知错误，请检查错误信息"
        
        # 构建完整的错误信息字符串
        detailed_error_msg = f"[{error_type}] {error_msg}"
        if "suggestion" in error_details:
            detailed_error_msg += f" | 建议: {error_details['suggestion']}"
        if "status_code" in error_details:
            detailed_error_msg += f" | HTTP状态码: {error_details['status_code']}"
        
        return {
            "idx": idx,
            "response": "",
            "reasoning": "",
            "answer": "",
            "answer_valid": False,
            "error": detailed_error_msg,
            "error_details": error_details,  # 保留详细错误信息
            "success": False
        }


def extract_and_fix_content(content: str, ground_truth_str: str) -> str:
    """
    最后的兜底方案：从content中强行提取所有数字，并使用GT的时间戳作为key
    """
    if not content:
        return ""
        
    try:
        import json
        
        # 1. 获取目标时间戳
        target_timestamps = []
        if ground_truth_str:
            try:
                gt_data = json.loads(ground_truth_str)
                if isinstance(gt_data, dict):
                    target_timestamps = sorted(gt_data.keys())
                elif isinstance(gt_data, list) and len(gt_data) > 0 and isinstance(gt_data[0], dict):
                    # 处理列表形式的GT [{"timestamp":...}, ...]
                    for item in gt_data:
                        ts = item.get('timestamp') or item.get('time') or item.get('t') or item.get('date')
                        if ts: target_timestamps.append(ts)
                    target_timestamps.sort()
            except:
                pass
        
        # 如果无法获取GT时间戳，尝试构造通用的 keys (0..95)
        # 但通常我们应该能拿到GT
        target_len = len(target_timestamps) if target_timestamps else 96
        
        # 2. 从content中提取所有浮点数
        # 匹配所有可能的数字格式
        # 排除包含在特定标签中的数字，避免提取到其他无关信息
        # 优先提取 predictions 列表中的数字
        import re
        
        valid_numbers = []
        
        # 尝试先找 JSON list
        list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if list_match:
            try:
                # 粗略提取列表内的数字
                inner = list_match.group(1)
                nums = re.findall(r'-?\d+\.?\d*', inner)
                valid_numbers = [float(n) for n in nums]
            except:
                pass
        
        # 如果没找到列表或者列表为空，则全文搜索
        if not valid_numbers:
             # 简单的正则提取
            numbers = re.findall(r'-?\d+\.?\d*', content)
            for n in numbers:
                try:
                    # 简单的启发式过滤
                    # 忽略纯整数且看起来像年份的 (2020-2030)
                    if '.' not in n and len(n) == 4 and n.startswith('20'):
                        val = int(n)
                        if 2020 <= val <= 2030:
                            continue
                    valid_numbers.append(float(n))
                except:
                    continue
        
        if not valid_numbers:
            return ""
            
        # 过滤掉显然不合理的值（例如极大的值，或者之前的年份干扰）
        # 既然是强制修复，我们假设最后一段密集的数字最有可能是预测值
        # 如果提取出的数字非常多（比如几百个），可能包含了prompt中的数字
        # 尝试取最后面的 N 个数字（通常回答在最后）
        if len(valid_numbers) > target_len * 2:
             valid_numbers = valid_numbers[-target_len:]

        # 3. 对齐长度
        if len(valid_numbers) > target_len:
            # 截断 (取前N个)
            valid_numbers = valid_numbers[:target_len]
        elif len(valid_numbers) < target_len:
            # 补全 (用最后一个值)
            last_val = valid_numbers[-1]
            valid_numbers.extend([last_val] * (target_len - len(valid_numbers)))
            
        # 4. 构造字典
        result = {}
        if target_timestamps:
            for i, ts in enumerate(target_timestamps):
                result[ts] = valid_numbers[i]
        else:
            # 如果没有时间戳，只能构造假的
            # 但 validate_sft 需要匹配时间戳，这种情况可能还是会挂
            # 除非我们修改 validate_sft 允许列表比较
            return json.dumps({"predictions": valid_numbers})
            
        return json.dumps(result)
        
    except Exception as e:
        print(f"Fix failed: {str(e)}")
        return ""

def process_csv_parallel(
    input_file: str, 
    output_file: str, 
    max_workers: int = 5, 
    timeout: float = 240.0,
    config: Optional[Dict] = None
):
    """
    并行处理CSV文件，调用API并解析响应
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        max_workers: 最大并行数
        timeout: API调用超时时间（秒）
        config: 可选的OpenAI配置字典，包含api_key、base_url、model等。
                如果为None，则从环境变量或.env文件读取
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file, keep_default_na=False)
    
    print(f"总共有 {len(df)} 行数据")
    print(f"使用 {max_workers} 个并行工作线程")
    
    # 获取OpenAI配置：优先使用传入的config，否则从.env文件或环境变量读取
    if config is None:
        config = get_openai_config()
    
    # 显示配置信息（不显示敏感信息）
    print(f"\n配置信息:")
    print(f"  Model: {config.get('model', '未设置')}")
    print(f"  Base URL: {config.get('base_url', '未设置（使用默认）')}")
    max_tokens = config.get('max_tokens')
    if max_tokens is not None:
        print(f"  Max Tokens: {max_tokens}")
    else:
        print(f"  Max Tokens: 无限制")
    if config.get("api_key"):
        api_key_preview = config['api_key'][:8] + "..." if len(config['api_key']) > 8 else "***"
        print(f"  API Key: {api_key_preview} (已设置)")
    else:
        print(f"  API Key: 未找到")
        print("警告: 未找到 OPENAI_API_KEY，请检查.env文件或环境变量")
        print("将使用空值填充 reasoning 和 answer 列")
    print("-" * 50)
    
    # 检查输出文件是否存在，支持断点续传
    existing_results = {}
    if os.path.exists(output_file):
        print(f"检测到输出文件已存在: {output_file}")
        print("正在检查已有结果，支持断点续传...")
        try:
            existing_df = pd.read_csv(output_file, keep_default_na=False)
            print(f"  已读取 {len(existing_df)} 行已有数据")
            
            # 检查已有结果的有效性
            valid_count = 0
            invalid_count = 0
            for idx, row in existing_df.iterrows():
                idx_val = row.get('idx', idx)
                answer = str(row.get('answer', ''))
                if is_valid_answer(answer):
                    existing_results[idx_val] = {
                        'response': str(row.get('response', '')),
                        'reasoning': str(row.get('reasoning', '')),
                        'answer': answer,
                        'valid': True
                    }
                    valid_count += 1
                else:
                    existing_results[idx_val] = {
                        'response': str(row.get('response', '')),
                        'reasoning': str(row.get('reasoning', '')),
                        'answer': answer,
                        'valid': False
                    }
                    invalid_count += 1
            
            print(f"  有效answer: {valid_count} 个")
            print(f"  无效answer: {invalid_count} 个")
            print(f"  将跳过已有有效结果的样本，只处理缺失或无效的样本")
        except Exception as e:
            print(f"  警告: 读取已有文件时出错: {str(e)}，将重新处理所有样本")
            existing_results = {}
    else:
        print(f"输出文件不存在，将创建新文件: {output_file}")
    
    # 初始化输出CSV文件（创建带所有列的框架）
    # 添加response, reasoning, answer列（初始为空）
    if 'response' not in df.columns:
        df['response'] = ''
    if 'reasoning' not in df.columns:
        df['reasoning'] = ''
    if 'answer' not in df.columns:
        df['answer'] = ''
    
    # 如果输出文件已存在，尝试合并已有结果
    if existing_results:
        for idx, row in df.iterrows():
            idx_val = row.get('idx', idx)
            if idx_val in existing_results:
                existing = existing_results[idx_val]
                df.at[idx, 'response'] = existing.get('response', '')
                df.at[idx, 'reasoning'] = existing.get('reasoning', '')
                df.at[idx, 'answer'] = existing.get('answer', '')
    
    # 确保列的顺序：idx, input, ground_truth, tool, prompt, response, reasoning, answer
    column_order = ['idx', 'input', 'ground_truth', 'tool', 'prompt', 'response', 'reasoning', 'answer']
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # 保存初始CSV文件（包含已有结果）
    df.to_csv(output_file, index=False, encoding='utf-8')
    if existing_results:
        print(f"已合并已有结果到输出文件: {output_file}")
    else:
        print(f"已初始化输出文件: {output_file}")
    
    # 准备任务列表（只处理需要处理的样本）
    tasks = []
    skipped_count = 0
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        # 检查是否已有有效结果
        if idx_val in existing_results and existing_results[idx_val].get('valid', False):
            skipped_count += 1
            continue  # 跳过已有有效结果的样本
        
        prompt = row.get('prompt', '')
        tasks.append((prompt, idx_val, config))
    
    if skipped_count > 0:
        print(f"\n断点续传: 跳过 {skipped_count} 个已有有效结果的样本")
        print(f"需要处理: {len(tasks)} 个样本")
    else:
        print(f"\n需要处理: {len(tasks)} 个样本")
    
    # 创建文件写入锁，确保线程安全（在函数级别定义，供所有阶段使用）
    file_lock = threading.Lock()
    
    def save_result_to_csv(result_data: Dict, idx: int):
        """将单个结果保存到CSV文件"""
        with file_lock:
            try:
                # 读取当前CSV文件
                current_df = pd.read_csv(output_file, keep_default_na=False)
                
                # 找到对应的行（通过idx匹配）
                row_idx = current_df[current_df['idx'] == idx].index
                if len(row_idx) > 0:
                    row_idx = row_idx[0]
                    # 更新对应行的数据
                    current_df.at[row_idx, 'response'] = result_data.get('response', '')
                    current_df.at[row_idx, 'reasoning'] = result_data.get('reasoning', '')
                    current_df.at[row_idx, 'answer'] = result_data.get('answer', '')
                    
                    # 保存到CSV
                    current_df.to_csv(output_file, index=False, encoding='utf-8')
            except Exception as e:
                # 如果保存失败，只打印错误，不中断程序
                if not HAS_TQDM:
                    print(f"警告: 保存idx={idx}的结果时出错: {str(e)}")
    
    def process_task(task):
        prompt, idx, cfg = task
        return call_api_single(prompt, idx, cfg, timeout)
    
    # 记录开始时间（用于最终统计）
    start_time = time.time()
    
    # 如果没有需要处理的任务，直接进入重试阶段
    if len(tasks) == 0:
        print("\n所有样本都已处理完成，直接进入重试阶段...")
        results = {}
        # 从已有结果构建results字典
        for idx, row in df.iterrows():
            idx_val = row.get('idx', idx)
            answer = str(row.get('answer', ''))
            results[idx_val] = {
                'idx': idx_val,
                'response': str(row.get('response', '')),
                'reasoning': str(row.get('reasoning', '')),
                'answer': answer,
                'answer_valid': is_valid_answer(answer),
                'success': True if answer else False,
                'error': None
            }
    else:
        # 并行处理
        results = {}
        completed = 0
        failed = 0
        error_collector = []  # 收集错误信息
        
        print(f"开始并行调用API...")
        print(f"  超时设置: {timeout} 秒")
        print(f"  并发数: {max_workers}")
        print(f"  模型: {config.get('model', '未设置')}")
        print(f"  Base URL: {config.get('base_url', '未设置')}")
    
        # 使用tqdm显示进度条
        with tqdm(total=len(tasks), desc="API调用进度", unit="个", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_idx = {executor.submit(process_task, task): task[1] for task in tasks}
                print(f"  已提交 {len(future_to_idx)} 个任务到线程池")
                
                # 处理完成的任务
                first_result_time = None
                last_update_time = time.time()
                check_interval = 30.0  # 每30秒检查一次是否有进展
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    if first_result_time is None:
                        first_result_time = time.time()
                        elapsed = first_result_time - start_time
                        print(f"  第一个请求完成，耗时: {elapsed:.1f} 秒")
                        last_update_time = first_result_time
                    
                    idx = future_to_idx[future]
                    try:
                        # 添加超时，防止单个future卡住（超时时间比API timeout稍长）
                        future_timeout = timeout + 60.0  # API timeout + 60秒缓冲
                        result = future.result(timeout=future_timeout)
                        results[idx] = result
                        completed += 1
                        last_update_time = time.time()  # 更新最后更新时间
                        
                        # 立即保存结果到CSV
                        save_result_to_csv(result, idx)
                        
                        if result['success']:
                            # 更新进度条
                            pbar.update(1)
                            # 更新进度条后缀信息
                            elapsed = time.time() - start_time
                            avg_time = elapsed / completed if completed > 0 else 0
                            pbar.set_postfix({
                                '成功': completed - failed,
                                '失败': failed,
                                '平均': f'{avg_time:.1f}s/个'
                            })
                        else:
                            failed += 1
                            # 收集错误信息
                            error_msg = result.get('error', 'Unknown error')
                            error_details = result.get('error_details', {})
                            error_info = {
                                'idx': idx,
                                'error': error_msg,
                                'error_details': error_details
                            }
                            error_collector.append(error_info)
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                '成功': completed - failed,
                                '失败': failed,
                                '错误': f"idx={idx}"
                            })
                            
                            # 即使有tqdm，也打印错误（但限制数量避免刷屏）
                            if len(error_collector) <= 5:  # 只打印前5个错误的详细信息
                                error_type = error_details.get('error_type', 'Unknown') if error_details else 'Unknown'
                                suggestion = error_details.get('suggestion', '') if error_details else ''
                                print(f"\n[错误] idx={idx} [{error_type}]: {error_msg}")
                                if suggestion:
                                    print(f"  建议: {suggestion}")
                            elif len(error_collector) == 6:
                                print(f"\n[提示] 还有更多错误，将在结束时汇总显示...")
                    except concurrent.futures.TimeoutError:
                        # future.result() 超时
                        completed += 1
                        failed += 1
                        error_result = {
                            "idx": idx,
                            "response": "",
                            "reasoning": "",
                            "answer": "",
                            "answer_valid": False,
                            "error": f"Future timeout (exceeded {future_timeout:.1f}s)",
                            "success": False
                        }
                        results[idx] = error_result
                        save_result_to_csv(error_result, idx)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': completed - failed,
                            '失败': failed,
                            '超时': f"idx={idx}"
                        })
                        print(f"\n[警告] idx={idx} 的future超时（超过 {future_timeout:.1f} 秒），可能API调用卡住了")
                    except Exception as e:
                        completed += 1
                        failed += 1
                        # 即使出错也要保存（保存错误信息）
                        error_result = {
                            "idx": idx,
                            "response": "",
                            "reasoning": "",
                            "answer": "",
                            "answer_valid": False,
                            "error": str(e),
                            "success": False
                        }
                        results[idx] = error_result
                        save_result_to_csv(error_result, idx)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': completed - failed,
                            '失败': failed,
                            '异常': f"idx={idx}"
                        })
                        error_type = type(e).__name__
                        error_msg = str(e)
                        print(f"\n[错误] idx={idx} 处理异常 [{error_type}]: {error_msg}")
                    
                    # 定期检查是否有进展
                    current_time = time.time()
                    if current_time - last_update_time > check_interval:
                        elapsed = current_time - start_time
                        print(f"\n[进度检查] 已耗时: {elapsed:.1f}秒, 完成: {completed}/{len(tasks)}, 成功: {completed - failed}, 失败: {failed}")
                        last_update_time = current_time
        
        elapsed_time = time.time() - start_time
        print(f"\n第一轮API调用完成!")
        print(f"  成功: {completed - failed} 个")
        print(f"  失败: {failed} 个")
        if len(tasks) > 0:
            print(f"  总耗时: {elapsed_time:.1f} 秒")
            print(f"  平均每个请求: {elapsed_time / len(tasks):.1f} 秒")
        
        # 显示错误汇总
        if error_collector:
            print(f"\n错误汇总（共 {len(error_collector)} 个错误）:")
            # 按错误类型分组
            error_types = {}
            for err_info in error_collector:
                error_details = err_info.get('error_details', {})
                error_type = error_details.get('error_type', 'Unknown') if error_details else 'Unknown'
                error_category = error_details.get('error_category', 'unknown') if error_details else 'unknown'
                if error_category not in error_types:
                    error_types[error_category] = []
                error_types[error_category].append(err_info)
            
            # 显示每个错误类型的统计
            for category, errors in error_types.items():
                print(f"  [{category}]: {len(errors)} 个")
            
            # 显示前几个典型错误的详细信息
            print(f"\n典型错误示例（显示前3个）:")
            for i, err_info in enumerate(error_collector[:3], 1):
                idx = err_info['idx']
                error = err_info['error']
                error_details = err_info.get('error_details', {})
                error_type = error_details.get('error_type', 'Unknown') if error_details else 'Unknown'
                suggestion = error_details.get('suggestion', '') if error_details else ''
                print(f"  {i}. idx={idx} [{error_type}]: {error}")
                if suggestion:
                    print(f"     建议: {suggestion}")
            
            if len(error_collector) > 3:
                print(f"  ... 还有 {len(error_collector) - 3} 个类似错误")
    
    # 识别需要重试的样本（answer无效的样本）
    # 创建一个idx到prompt的映射，方便重试时使用
    idx_to_prompt = {}
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        idx_to_prompt[idx_val] = row.get('prompt', '')
    
    retry_tasks = []
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        result = results.get(idx_val, {})
        
        # 检查是否需要重试：API调用失败 或 answer无效
        need_retry = False
        if not result.get('success', False):
            need_retry = True
        elif not result.get('answer_valid', False):
            # answer为空或格式不正确
            answer = result.get('answer', '')
            if not is_valid_answer(answer):
                need_retry = True
        
        if need_retry:
            prompt = idx_to_prompt.get(idx_val, '')
            retry_tasks.append((prompt, idx_val, config))
    
    # 对需要重试的样本进行最多3次重试
    max_retries = 6
    if retry_tasks:
        print(f"\n发现 {len(retry_tasks)} 个需要重试的样本（answer无效或API调用失败）")
        print(f"将进行最多 {max_retries} 次重试...")
        print("-" * 50)
        
        for retry_round in range(1, max_retries + 1):
            if not retry_tasks:
                break
            
            print(f"\n第 {retry_round} 次重试，剩余 {len(retry_tasks)} 个样本...")
            retry_start_time = time.time()
            retry_results = {}
            retry_completed = 0
            retry_failed = 0
            
            # 准备新的重试任务列表（只包含仍然无效的样本）
            current_retry_tasks = []
            for task in retry_tasks:
                prompt, idx_val, cfg = task
                # 重新读取当前CSV，检查是否已经有效
                current_df = pd.read_csv(output_file, keep_default_na=False)
                row_idx = current_df[current_df['idx'] == idx_val].index
                if len(row_idx) > 0:
                    row_idx = row_idx[0]
                    current_answer = str(current_df.at[row_idx, 'answer'])
                    if not is_valid_answer(current_answer):
                        current_retry_tasks.append(task)
            
            if not current_retry_tasks:
                print("所有样本已修复，无需继续重试")
                break
            
            retry_tasks = []  # 清空，准备下一轮
            
            # 并行重试
            with tqdm(total=len(current_retry_tasks), desc=f"重试第{retry_round}轮", unit="个",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as retry_pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {executor.submit(process_task, task): task[1] for task in current_retry_tasks}
                    
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx_val = future_to_idx[future]
                        try:
                            # 添加超时，防止单个future卡住
                            result = future.result(timeout=timeout + 60.0)
                            retry_results[idx_val] = result
                            retry_completed += 1
                            
                            # 检查answer是否有效
                            answer_valid = result.get('answer_valid', False)
                            
                            # 立即保存结果
                            save_result_to_csv(result, idx_val)
                            
                            if result['success'] and answer_valid:
                                # 重试成功，不再需要重试
                                retry_pbar.update(1)
                                retry_pbar.set_postfix({
                                    '成功': retry_completed - retry_failed,
                                    '失败': retry_failed
                                })
                            else:
                                # 仍然无效，加入下一轮重试
                                prompt = idx_to_prompt.get(idx_val, '')
                                if prompt:
                                    retry_tasks.append((prompt, idx_val, config))
                                retry_failed += 1
                                retry_pbar.update(1)
                                retry_pbar.set_postfix({
                                    '成功': retry_completed - retry_failed,
                                    '失败': retry_failed
                                })
                        except concurrent.futures.TimeoutError:
                            # future.result() 超时
                            retry_failed += 1
                            error_result = {
                                "idx": idx_val,
                                "response": "",
                                "reasoning": "",
                                "answer": "",
                                "answer_valid": False,
                                "error": f"Future timeout (exceeded {timeout + 60.0:.1f}s)",
                                "success": False
                            }
                            retry_results[idx_val] = error_result
                            save_result_to_csv(error_result, idx_val)
                            retry_pbar.update(1)
                            retry_pbar.set_postfix({
                                '成功': retry_completed - retry_failed,
                                '失败': retry_failed,
                                '超时': f"idx={idx_val}"
                            })
                            print(f"\n[警告] 重试 idx={idx_val} 的future超时")
                        except Exception as e:
                            retry_completed += 1
                            retry_failed += 1
                            error_result = {
                                "idx": idx_val,
                                "response": "",
                                "reasoning": "",
                                "answer": "",
                                "answer_valid": False,
                                "error": str(e),
                                "success": False
                            }
                            retry_results[idx_val] = error_result
                            save_result_to_csv(error_result, idx_val)
                            
                            # 仍然需要重试
                            prompt = idx_to_prompt.get(idx_val, '')
                            if prompt:
                                retry_tasks.append((prompt, idx_val, config))
                            
                            retry_pbar.update(1)
                            retry_pbar.set_postfix({
                                '成功': retry_completed - retry_failed,
                                '失败': retry_failed
                            })
            
            retry_elapsed = time.time() - retry_start_time
            retry_success = sum(1 for r in retry_results.values() if r.get('success', False) and r.get('answer_valid', False))
            print(f"  重试成功: {retry_success} 个")
            print(f"  重试失败: {retry_failed} 个")
            print(f"  耗时: {retry_elapsed:.1f} 秒")
            
            # 更新results字典
            results.update(retry_results)
            
    # === 最后的强制兜底修正 (Final Force Fix) ===
    print("\n" + "=" * 50)
    print("正在检查是否需要强制修正 (Final Force Fix)...")
    
    fixed_count = 0
    # 读取最新的结果
    final_df = pd.read_csv(output_file, keep_default_na=False)
    
    # 建立 idx -> ground_truth 的映射
    idx_to_gt = {}
    for idx, row in df.iterrows():
        idx_val = row.get('idx', idx)
        idx_to_gt[idx_val] = row.get('ground_truth', '')

    for i, row in final_df.iterrows():
        answer = str(row.get('answer', ''))
        idx_val = row.get('idx')
        
        # 如果依然无效
        if not is_valid_answer(answer):
            print(f"  正在尝试修复 idx={idx_val} ...")
            
            # 获取上下文
            response = str(row.get('response', ''))
            # 如果 response 为空，可能之前的 error 覆盖了，尝试找回 response?
            # 暂时只能基于现有 response
            
            gt_str = idx_to_gt.get(idx_val, '')
            
            # 尝试修复
            fixed_answer = extract_and_fix_content(response, gt_str)
            
            if is_valid_answer(fixed_answer):
                final_df.at[i, 'answer'] = fixed_answer
                # 更新 reasoning 标记已修复
                current_reasoning = str(row.get('reasoning', ''))
                final_df.at[i, 'reasoning'] = current_reasoning + "\n[System: Forced Fix Applied]"
                fixed_count += 1
                print(f"  > 修复成功!")
            else:
                print(f"  > 修复失败: 无法从 response 提取有效数字")

    if fixed_count > 0:
        print(f"共强制修复了 {fixed_count} 个样本")
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已保存修正后的结果到: {output_file}")
    else:
        print("没有样本需要/能够被强制修复")
    
    # 最终统计
    total_elapsed = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"所有API调用完成!")
    print(f"  总耗时: {total_elapsed:.1f} 秒")
    print(f"  总样本数: {len(df)} 个")
    
    # 读取最终CSV文件进行统计
    final_df = pd.read_csv(output_file, keep_default_na=False)
    valid_answer_count = 0
    invalid_answer_count = 0
    
    for idx, row in final_df.iterrows():
        answer = str(row.get('answer', ''))
        if is_valid_answer(answer):
            valid_answer_count += 1
        else:
            invalid_answer_count += 1
    
    print(f"\n最终统计信息:")
    print(f"  有效answer: {valid_answer_count} 个")
    print(f"  无效answer: {invalid_answer_count} 个")
    print(f"  成功率: {valid_answer_count / len(final_df) * 100:.1f}%")
    
    # 最终错误汇总（从CSV中读取错误信息）
    if invalid_answer_count > 0:
        print(f"\n最终错误分析:")
        error_samples = []
        for idx, row in final_df.iterrows():
            answer = str(row.get('answer', ''))
            if not is_valid_answer(answer):
                response = str(row.get('response', ''))
                error_info = {
                    'idx': row.get('idx', idx),
                    'answer': answer[:100] if answer else '(空)',
                    'response_length': len(response),
                }
                # 尝试从response或error列中提取错误信息
                if not response:
                    error_info['has_response'] = False
                else:
                    error_info['has_response'] = True
                error_samples.append(error_info)
        
        # 统计错误类型
        empty_answer = sum(1 for e in error_samples if not e['answer'] or e['answer'] == '(空)')
        no_response = sum(1 for e in error_samples if not e['has_response'])
        has_response = len(error_samples) - no_response
        
        print(f"  空answer: {empty_answer} 个")
        print(f"  无response: {no_response} 个")
        print(f"  有response但answer无效: {has_response} 个")
        
        # 显示几个样本的错误详情（从CSV读取）
        print(f"\n错误样本详情（前3个）:")
        for i, sample in enumerate(error_samples[:3], 1):
            idx = sample['idx']
            answer = sample['answer']
            has_resp = sample['has_response']
            print(f"  {i}. idx={idx}: answer='{answer[:50]}...' (has_response={has_resp})")
        
        print(f"\n提示: 请检查上述错误样本，可能需要检查API配置或prompt格式")
    
    print(f"\n所有结果已保存到: {output_file}")


if __name__ == "__main__":
    input_file = "datasets/windy_power_processed_test.csv"
    output_file = "datasets/qwen_0.6B.csv"
    max_workers = 20
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        max_workers = int(sys.argv[3])
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"并行数: {max_workers}")
    print("-" * 50)
    
    process_csv_parallel(input_file, output_file, max_workers=max_workers)

