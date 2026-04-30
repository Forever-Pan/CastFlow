from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

import agentlightning as agl
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .rewards import calculate_mse, compute_contrastive_reward, extract_answer_from_tags, parse_json_column


RL_PROMPT_SUFFIX = (
    "You can output your response more flexible(no longer strictly based on Reference model! "
    "You can use other tool and your knowledge! But be short)"
)


class LiteAgent(agl.LitAgent[dict[str, Any]]):
    """AgentLightning rollout agent migrated from the original rl_agent.py.

    The training semantics are intentionally close to the original script:
    each rollout calls the hosted vLLM OpenAI-compatible endpoint supplied by
    AgentLightning, appends the same RL prompt suffix, computes the contrastive
    reward against `model_auxiliary_tool`, and returns the scalar reward to VERL.
    """

    _task_counter = 0

    def __init__(self, rollout_output_dir: Optional[str] = None) -> None:
        super().__init__()
        self.rollout_output_dir = Path(rollout_output_dir) if rollout_output_dir else None
        if self.rollout_output_dir is not None:
            self.rollout_output_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_to_idx: dict[int, int] = {}

    def rollout(
        self,
        task: dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float:
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        attempted_rollout = cast(agl.AttemptedRollout, rollout)
        base_url = llm.get_base_url(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id)

        chat_model = init_chat_model(
            model="hosted_vllm/" + llm.model,
            model_provider="openai",
            openai_api_base=base_url,
            openai_api_key=llm.api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=llm.sampling_parameters.get("temperature", 0.0),
        )

        prompt = str(task["prompt"]) + RL_PROMPT_SUFFIX
        dataset_name = str(task.get("dataset_name", ""))
        messages = [HumanMessage(content=prompt)]

        handler = self.tracer.get_langchain_handler()
        if handler:
            from langchain_core.runnables import RunnableConfig

            result = chat_model.invoke(messages, config=RunnableConfig(callbacks=[handler]))
        else:
            result = chat_model.invoke(messages)

        answer = result.content if hasattr(result, "content") else str(result)
        ground_truth = str(task.get("ground_truth", ""))
        tool = task.get("tool", "")

        answer_content = extract_answer_from_tags(str(answer))
        answer_dict = parse_json_column(answer_content)
        gt_dict = parse_json_column(ground_truth)
        mse = calculate_mse(answer_dict, gt_dict)
        reward = compute_contrastive_reward(str(answer), ground_truth, dataset_name, tool=tool)

        if reward is None or np.isnan(reward) or np.isinf(reward):
            reward = -1.0

        if self.rollout_output_dir is not None:
            task_idx = self._get_task_idx(task, prompt)
            self._save_rollout(
                rollout_id=attempted_rollout.rollout_id,
                prompt=prompt,
                answer=str(answer),
                ground_truth=ground_truth,
                reward=float(reward),
                mse=float(mse) if not np.isnan(mse) else None,
                task_idx=task_idx,
            )

        return float(reward)

    def _get_task_idx(self, task: dict[str, Any], prompt: str) -> int:
        if "idx" in task and task["idx"] is not None:
            try:
                return int(task["idx"]) + 1
            except (TypeError, ValueError):
                pass

        prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        prompt_hash_key = int(prompt_md5, 16)
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
        if self.rollout_output_dir is None:
            return
        prompt_dir = self.rollout_output_dir / f"prompt_{task_idx}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        filepath = prompt_dir / (rollout_id.replace("ro-", "") + ".json")
        payload = {
            "rollout_id": rollout_id,
            "prompt": prompt,
            "answer": answer,
            "ground_truth": ground_truth,
            "reward": float(reward) if not (np.isnan(reward) or np.isinf(reward)) else None,
            "mse": float(mse) if mse is not None and not np.isnan(mse) and not np.isinf(mse) else None,
        }
        with filepath.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
