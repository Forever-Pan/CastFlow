"""Canonical forecasting chat contexts shared by inference and training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def build_forecast_messages(
    prompt: str, system_prompt: str = ""
) -> list[dict[str, str]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("forecasting context requires a nonempty text prompt")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be text")
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def resolve_forecast_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    system = task.get("system_prompt", "")
    expected = build_forecast_messages(task["prompt"], "" if system is None else system)
    supplied = task.get("messages")
    if supplied is None or isinstance(supplied, str) and not supplied.strip():
        return expected
    if isinstance(supplied, str):
        try:
            supplied = json.loads(supplied)
        except json.JSONDecodeError as exc:
            raise ValueError("messages must be a JSON system/user chat list") from exc
    if hasattr(supplied, "tolist"):
        supplied = supplied.tolist()
    if not isinstance(supplied, list) or any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or not isinstance(message["role"], str)
        or not isinstance(message["content"], str)
        for message in supplied
    ):
        raise ValueError("messages must contain only text role/content chat entries")
    if supplied != expected:
        raise ValueError("stored messages disagree with prompt/system_prompt")
    return expected
