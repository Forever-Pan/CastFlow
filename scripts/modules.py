from __future__ import annotations

import json
import re

import numpy as np

from .config import CastFlowConfig
from .evaluation import parse_forecast_value
from .llm import OpenAICompatibleLLM
from .prompts import build_api_forecasting_messages
from .schema import ForecastResult, ToolObservation, WorkflowState
from .toolkit import MultiViewToolkit, round_list


class PlanningModule:
    """Frozen general-purpose planner for tool scheduling."""

    mandatory_tools = ["model_auxiliary_tool", "exogenous_analysis_tool"]

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def plan(self, state: WorkflowState, config: CastFlowConfig) -> list[str]:
        api_plan = self._api_plan(state, config)
        if api_plan:
            return api_plan
        memory_tools = self._tools_from_memory(state)
        planned = list(self.mandatory_tools)
        planned.extend(tool for tool in memory_tools if tool not in planned)
        planned.extend(tool for tool in self._heuristic_tools(state) if tool not in planned)
        return planned

    def _api_plan(self, state: WorkflowState, config: CastFlowConfig) -> list[str]:
        if self.llm is None or not self.llm.available():
            return []
        optional_tools = [
            "statistical_analysis_tool",
            "basic_statistics_tool",
            "data_quality_tool",
            "comprehensive_feature_tool",
            "trend_analysis_tool",
            "changepoint_trend_tool",
            "cross_channel_tool",
            "event_summary_tool",
        ]
        if config.allow_train_only_tools:
            optional_tools.append("autoregressive_residual_tool")

        user = json.dumps(
            {
                "task": "Select CastFlow diagnostic tools for leakage-free time-series forecasting.",
                "mandatory_tools": self.mandatory_tools,
                "optional_tools": optional_tools,
                "lookback": config.lookback,
                "horizon": config.horizon,
                "target_tail": round_list(state.window.target_history[-min(24, len(state.window.target_history)) :]),
                "retrieved_memory_tool_schedules": [
                    item.get("tool_schedule", []) for item in state.retrieved_memory[: config.memory_top_k]
                ],
                "output_contract": {"tools": ["tool_name_1", "tool_name_2"]},
            },
            ensure_ascii=False,
        )
        response = self.llm.chat(
            system=(
                "You are the frozen CastFlow planning module. Return only JSON. "
                "Always include mandatory tools and choose useful optional tools from the given list."
            ),
            user=user,
            max_tokens=config.planner_max_tokens,
            temperature=0.0,
        )
        if not response.ok:
            state.reflection = append_note(state.reflection, f"[planner_api_fallback] {response.error}")
            return []

        selected = parse_tool_schedule(response.content)
        allowed = set(self.mandatory_tools + optional_tools)
        planned = list(self.mandatory_tools)
        planned.extend(tool for tool in selected if tool in allowed and tool not in planned)
        return planned if len(planned) > len(self.mandatory_tools) else []

    def _tools_from_memory(self, state: WorkflowState) -> list[str]:
        tools: list[str] = []
        for item in state.retrieved_memory:
            memory_tools = item.get("tool_schedule") or [
                modern_tool_name(call.get("tool_name", ""))
                for call in item.get("tool_calls", [])
                if isinstance(call, dict)
            ]
            for tool in memory_tools:
                if tool not in tools:
                    tools.append(tool)
        return tools

    def _heuristic_tools(self, state: WorkflowState) -> list[str]:
        return ["statistical_analysis_tool", "trend_analysis_tool", "changepoint_trend_tool"]


class ActionModule:
    """Deterministic execution interface for the multi-view toolkit."""

    def __init__(self, toolkit: MultiViewToolkit | None = None) -> None:
        self.toolkit = toolkit or MultiViewToolkit()

    def execute(self, state: WorkflowState, config: CastFlowConfig) -> list[ToolObservation]:
        observations = [self.toolkit.execute(name, state.window, config) for name in state.tool_schedule]
        aux = next((obs.output for obs in observations if obs.tool_name == "model_auxiliary_tool"), None)
        if aux:
            state.baseline_forecast = aux.get("ensemble_forecast_baseline")
        return observations


class ForecastingModule:
    """Specialized numerical forecaster backed by the configured API."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def forecast(self, state: WorkflowState, config: CastFlowConfig) -> list[float]:
        if self.llm is not None and self.llm.available():
            api_forecast = self._api_forecast(state, config)
            if api_forecast is not None:
                return api_forecast
            return invalid_forecast(config)
        api_forecast = self._api_forecast(state, config)
        if api_forecast is not None:
            return api_forecast
        return self._deterministic_forecast(state, config)

    def _api_forecast(self, state: WorkflowState, config: CastFlowConfig) -> list[float] | None:
        if self.llm is None or not self.llm.available():
            return None
        observations = state.observation_map()
        system, user = build_api_forecasting_messages(
            history=state.window.history,
            future=state.window.future if config.allow_train_only_tools else None,
            target_col=state.window.target_col,
            timestamp_col=state.window.timestamp_col,
            horizon=config.horizon,
            lookback=config.lookback,
            tool_outputs=observations,
            tool_schedule=state.tool_schedule,
            retry_feedback=state.reflection,
        )
        state.forecast_full_prompt = user
        response = self.llm.chat(
            system=system,
            user=user,
            max_tokens=config.forecasting_max_tokens,
            temperature=0.3,
        )
        if not response.ok:
            state.reflection = append_note(state.reflection, f"[forecast_api_fallback] {response.error}")
            return None
        state.forecast_full_response = response.content

        if not re.search(r"<answer>(.*?)</answer>", response.content, flags=re.DOTALL | re.IGNORECASE):
            state.reflection = append_note(state.reflection, "[forecast_api_fallback] missing <answer> content")
            return None
        parsed = parse_forecast_value(response.content)
        if not parsed:
            state.reflection = append_note(state.reflection, "[forecast_api_fallback] empty or unparsable answer")
            return None
        ordered = [parsed[key] for key in sorted(parsed, key=forecast_key_sort)]
        if len(ordered) != config.horizon:
            state.reflection = append_note(
                state.reflection,
                f"[forecast_api_fallback] answer length {len(ordered)} != horizon {config.horizon}",
            )
            return None
        return round_list(ordered)

    def _deterministic_forecast(self, state: WorkflowState, config: CastFlowConfig) -> list[float]:
        observations = state.observation_map()
        if not state.forecast_full_prompt:
            _, prompt = build_api_forecasting_messages(
                history=state.window.history,
                future=state.window.future if config.allow_train_only_tools else None,
                target_col=state.window.target_col,
                timestamp_col=state.window.timestamp_col,
                horizon=config.horizon,
                lookback=config.lookback,
                tool_outputs=observations,
                tool_schedule=state.tool_schedule,
                retry_feedback=state.reflection,
            )
            state.forecast_full_prompt = prompt
        baseline = observations.get("model_auxiliary_tool", {}).get("ensemble_forecast_baseline")
        if baseline is not None and config.allow_train_only_tools:
            rounded = round_list(np.asarray(baseline, dtype=float))
            state.forecast_full_response = f"<think>Fallback to model_auxiliary_tool due to API or parsing failure.</think>\n<answer>{json.dumps({str(i): value for i, value in enumerate(rounded)}, ensure_ascii=False)}</answer>"
            return rounded
        if baseline is None:
            baseline = [float(state.window.target_history[-1])] * config.horizon

        forecast = np.asarray(baseline, dtype=float)
        trend = observations.get("trend_analysis_tool", {})
        changepoint = observations.get("changepoint_trend_tool", {})
        event = observations.get("event_summary_tool", {})
        quality = observations.get("data_quality_tool", {})

        slope = float(trend.get("slope", 0.0) or 0.0)
        recent_momentum = float(changepoint.get("recent_momentum", 0.0) or 0.0)
        steps = np.arange(1, len(forecast) + 1, dtype=float)
        forecast = forecast + 0.20 * (0.7 * slope + 0.3 * recent_momentum) * steps

        pattern = event.get("dominant_pattern")
        if pattern == "flat":
            level = float(np.mean(state.window.target_history[-min(12, len(state.window.target_history)) :]))
            forecast = 0.85 * forecast + 0.15 * level
        elif pattern == "oscillation":
            forecast = 0.95 * forecast + 0.05 * np.mean(state.window.target_history)

        boundary = quality.get("clipping_boundary")
        if isinstance(boundary, list) and len(boundary) == 2:
            forecast = np.clip(forecast, float(boundary[0]), float(boundary[1]))

        rounded = round_list(forecast)
        state.forecast_full_response = f"<think>Deterministic CastFlow fallback.</think>\n<answer>{json.dumps({str(i): value for i, value in enumerate(rounded)}, ensure_ascii=False)}</answer>"
        return rounded


class ReflectionModule:
    """Frozen reflection and deterministic quality gatekeeper."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def validate(
        self,
        state: WorkflowState,
        config: CastFlowConfig,
        metrics: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        if state.forecast is None:
            return False, append_note(state.reflection, "forecast is missing")
        if len(state.forecast) != config.horizon:
            return False, append_note(state.reflection, f"forecast length {len(state.forecast)} != horizon {config.horizon}")
        values = np.asarray(state.forecast, dtype=float)
        if not np.all(np.isfinite(values)):
            return False, append_note(state.reflection, "forecast contains non-finite values")
        api_message = self._api_reflect(state, config, metrics=metrics)
        if api_message:
            if "FAIL" in api_message.upper() and "PASS" not in api_message.upper():
                return False, api_message
            return True, api_message
        return True, "PASS"

    def _api_reflect(
        self,
        state: WorkflowState,
        config: CastFlowConfig,
        metrics: dict[str, float] | None = None,
    ) -> str:
        if self.llm is None or not self.llm.available() or state.forecast is None:
            return ""
        payload = {
            "task": "Validate CastFlow forecast consistency.",
            "horizon": config.horizon,
            "tool_schedule": state.tool_schedule,
            "forecast": state.forecast,
            "baseline": state.baseline_forecast,
            "diagnostic_evidence": state.observation_map(),
            "prediction_metrics": metrics or {},
            "output_contract": "Return PASS or FAIL followed by one concise reason.",
        }
        response = self.llm.chat(
            system="You are the frozen CastFlow reflection module. Check format and evidence alignment.",
            user=json.dumps(payload, ensure_ascii=False),
            max_tokens=config.reflection_max_tokens,
            temperature=0.0,
        )
        if not response.ok:
            return f"FAIL [reflection_api_unavailable: {response.error}]"
        return response.content.strip() or "PASS"

    def to_result(self, state: WorkflowState) -> ForecastResult:
        if state.forecast is None:
            raise ValueError("cannot build result before forecast exists")
        return ForecastResult(
            forecast=state.forecast,
            baseline_forecast=state.baseline_forecast,
            tool_schedule=state.tool_schedule,
            observations=state.observation_map(),
            reflection=state.reflection,
            forecast_full_prompt=state.forecast_full_prompt,
            forecast_full_response=state.forecast_full_response,
        )


def parse_tool_schedule(text: str) -> list[str]:
    try:
        payload = json.loads(extract_json_object(text))
        tools = payload.get("tools", [])
        if isinstance(tools, list):
            return [str(tool) for tool in tools]
    except Exception:
        pass
    known = [
        "model_auxiliary_tool",
        "exogenous_analysis_tool",
        "statistical_analysis_tool",
        "basic_statistics_tool",
        "data_quality_tool",
        "comprehensive_feature_tool",
        "trend_analysis_tool",
        "changepoint_trend_tool",
        "cross_channel_tool",
        "event_summary_tool",
        "autoregressive_residual_tool",
    ]
    return [tool for tool in known if re.search(rf"\b{re.escape(tool)}\b", text)]


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return match.group(0)


def append_note(existing: str, note: str) -> str:
    return f"{existing}\n{note}".strip() if existing else note


def invalid_forecast(config: CastFlowConfig) -> list[float]:
    return [float("nan")] * config.horizon


def modern_tool_name(name: str) -> str:
    return {
        "statistical_analysis": "statistical_analysis_tool",
        "trend_analysis": "trend_analysis_tool",
        "exogenous_analysis": "exogenous_analysis_tool",
        "ar_residual_tool": "autoregressive_residual_tool",
    }.get(name, name)


def forecast_key_sort(key: str) -> tuple[int, object]:
    text = str(key)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)
