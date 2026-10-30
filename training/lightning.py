from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Any

from agentlightning.verl.daemon import AgentModeDaemon


class ForecastingDaemon(AgentModeDaemon):
    """Keep complete, single-response groups at the GRPO training boundary."""

    def _validate_completed(
        self,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ) -> None:
        completed = self._completed_rollouts_v0
        if not completed or len(completed) != self._total_tasks_queued:
            raise RuntimeError(
                "RLVR did not receive every queued response; check backend and rollout errors before restarting"
            )
        groups: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        for identifier, rollout in completed.items():
            reward = rollout.final_reward
            if (
                isinstance(reward, bool)
                or not isinstance(reward, Real)
                or not math.isfinite(reward)
            ):
                raise RuntimeError(
                    f"RLVR rollout {identifier} has no finite explicit reward; check the model request, reward, and output-writing errors"
                )
            if not rollout.triplets or len(rollout.triplets) != 1:
                raise RuntimeError(
                    f"RLVR rollout {identifier} requires exactly one token trace; check proxy token-ID collection and backend errors"
                )
            trace = rollout.triplets[0]
            prompt = trace.prompt.get("token_ids")
            response = trace.response.get("token_ids")
            for field, tokens in (("prompt", prompt), ("response", response)):
                if (
                    not isinstance(tokens, list)
                    or not tokens
                    or any(
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or token < 0
                        for token in tokens
                    )
                ):
                    raise RuntimeError(
                        f"RLVR rollout {identifier} has no valid {field} token IDs; check the tokenizer and proxy backend"
                    )
            if max_prompt_length is not None and len(prompt) > max_prompt_length:
                raise RuntimeError(
                    f"RLVR rollout {identifier} exceeds the prompt limit; contexts are not silently truncated"
                )
            if max_response_length is not None and len(response) > max_response_length:
                raise RuntimeError(
                    f"RLVR rollout {identifier} exceeds the response limit; check backend max_tokens handling"
                )
            source = self._task_id_to_original_sample.get(identifier)
            if not isinstance(source, dict) or not isinstance(
                source.get("data_id"), str
            ):
                raise RuntimeError(
                    f"RLVR rollout {identifier} has no original context identity"
                )
            groups[source["data_id"]].append(tuple(prompt))
        expected = self.train_rollout_n if self.is_train else 1
        for prompts in groups.values():
            if len(prompts) != expected or len(set(prompts)) != 1:
                raise RuntimeError(
                    "RLVR requires complete groups with identical prompt tokens; failed candidates cannot be dropped or replaced"
                )

    def get_train_data_batch(
        self,
        max_prompt_length: int,
        max_response_length: int,
        device: Any,
        global_steps: int,
    ) -> Any:
        self._validate_completed(max_prompt_length, max_response_length)
        return super().get_train_data_batch(
            max_prompt_length, max_response_length, device, global_steps
        )

    def get_test_metrics(self) -> Any:
        self._validate_completed()
        return super().get_test_metrics()
