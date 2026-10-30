from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

import agentlightning as agl
from openai import OpenAI

from training.rewards import (
    calculate_mse,
    compute_contrastive_reward,
    parse_json_column,
    serialize_forecast,
)
from utils.io import atomic_write_json


class LiteAgent(agl.LitAgent[dict[str, Any]]):
    """Score one forecasting continuation under a fixed evidence context.

    AgentLightning schedules the G independent responses for GRPO. A candidate
    does not acquire new tools or run the inference-time reflection/retry loop.
    """

    def __init__(
        self,
        rollout_output_dir: str | None = None,
        *,
        max_response_length: int = 5000,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 150,
        repetition_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.rollout_output_dir = (
            Path(rollout_output_dir) if rollout_output_dir else None
        )
        self.max_response_length = max_response_length
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.repetition_penalty = repetition_penalty

    def rollout(
        self, task: dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout
    ) -> float:
        llm = cast(agl.LLM, resources["main_llm"])
        attempt = cast(agl.AttemptedRollout, rollout)
        base_url = llm.get_base_url(attempt.rollout_id, attempt.attempt.attempt_id)
        prompt = task["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("RLVR task requires a nonempty prompt")
        # Transport failures are rollout failures, not low-reward model outputs.
        with OpenAI(
            base_url=base_url,
            api_key=llm.api_key or "EMPTY",
            max_retries=0,
            timeout=600,
        ) as client:
            response = client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm.sampling_parameters.get(
                    "temperature", self.temperature
                ),
                top_p=self.top_p,
                max_tokens=self.max_response_length,
                n=1,
                extra_body={
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty,
                    "return_token_ids": True,
                },
            )
        if not response.choices:
            raise RuntimeError("forecasting backend returned no completion choices")
        message = response.choices[0].message
        answer = message.content or ""
        if not isinstance(answer, str):
            raise TypeError("forecasting backend returned non-text completion content")
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning and "<think>" not in answer.lower():
            answer = f"<think>{reasoning}</think>\n{answer}"
        truth = serialize_forecast(task["ground_truth"])
        reward = compute_contrastive_reward(
            answer,
            truth,
            str(task["dataset_name"]),
            tool=task.get("tool"),
            baseline=task.get("baseline_forecast"),
        )
        if not math.isfinite(reward):
            raise RuntimeError("forecast reward must be finite")
        if self.rollout_output_dir is not None:
            mse = calculate_mse(parse_json_column(answer), parse_json_column(truth))
            task_id = hashlib.sha256(
                (str(task.get("dataset_name", "")) + "\n" + prompt).encode()
            ).hexdigest()[:20]
            attempt_id = hashlib.sha256(
                f"{attempt.rollout_id}:{attempt.attempt.attempt.attempt_id}".encode()
            ).hexdigest()[:24]
            atomic_write_json(
                self.rollout_output_dir / task_id / f"{attempt_id}.json",
                {
                    "rollout_id": attempt.rollout_id,
                    "attempt_id": attempt.attempt.attempt.attempt_id,
                    "prompt": prompt,
                    "answer": answer,
                    "ground_truth": truth,
                    "reward": float(reward),
                    "mse": float(mse) if math.isfinite(mse) else None,
                },
            )
        return float(reward)
