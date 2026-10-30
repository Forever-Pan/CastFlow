from __future__ import annotations

import json
import re

import numpy as np

from utils.config import CastFlowConfig
from utils.io import forecast_timestamps
from utils.llm import OpenAICompatibleLLM
from utils.schema import ForecastResult, ToolObservation, WorkflowState
from evaluation.evaluate import (
    cleanup_jsonish,
    extract_answer_text,
    parse_forecast_sequence,
)
from forecasting.prompts import (
    build_api_forecasting_messages,
    procedural_guidance,
    tool_description,
)
from forecasting.toolkit import MultiViewToolkit, round_list


class PlanningModule:
    """Frozen general-purpose planner for tool scheduling."""

    mandatory_tools = ["model_auxiliary_tool", "exogenous_analysis_tool"]

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def plan(self, state: WorkflowState, config: CastFlowConfig) -> list[str]:
        api_plan = self._api_plan(state, config)
        if api_plan is not None:
            return api_plan
        memory_tools = self._tools_from_memory(state)
        planned = list(self.mandatory_tools)
        allowed = set(self.mandatory_tools + optional_tools(config))
        planned.extend(
            tool for tool in memory_tools if tool in allowed and tool not in planned
        )
        planned.extend(
            tool for tool in self._heuristic_tools(state) if tool not in planned
        )
        return planned

    def _api_plan(
        self, state: WorkflowState, config: CastFlowConfig
    ) -> list[str] | None:
        if self.llm is None or not self.llm.available():
            return None
        allowed_optional = optional_tools(config)

        user = json.dumps(
            {
                "task": "Select CastFlow diagnostic tools for leakage-free time-series forecasting.",
                "mandatory_tools": self.mandatory_tools,
                "optional_tools": allowed_optional,
                "tool_descriptions": {
                    name: tool_description(name)
                    for name in self.mandatory_tools + allowed_optional
                },
                "lookback": config.lookback,
                "horizon": config.horizon,
                "target_tail": round_list(
                    state.window.target_history[
                        -min(24, len(state.window.target_history)) :
                    ]
                ),
                "historical_procedural_examples": procedural_guidance(
                    state.retrieved_memory[: config.memory_top_k]
                ),
                "previous_attempt_feedback": state.reflection,
                "request_history": state.history,
                "output_contract": {"tools": ["tool_name_1", "tool_name_2"]},
            },
            ensure_ascii=False,
            sort_keys=True,
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
        if not response.ok or getattr(response, "finish_reason", None) in {
            "length",
            "content_filter",
        }:
            state.reflection = append_note(
                state.reflection,
                f"[planner_api_fallback] {response.error or 'response truncated or filtered'}",
            )
            return None

        selected = parse_tool_schedule(response.content)
        allowed = set(self.mandatory_tools + allowed_optional)
        if not selected:
            try:
                if json.loads(extract_json_object(response.content)).get("tools") != []:
                    return None
            except (ValueError, TypeError, AttributeError):
                return None
        planned = list(self.mandatory_tools)
        planned.extend(
            tool for tool in selected if tool in allowed and tool not in planned
        )
        return planned

    def _tools_from_memory(self, state: WorkflowState) -> list[str]:
        tools: list[str] = []
        for item in state.retrieved_memory:
            memory_tools = item.get("tool_schedule") or [
                modern_tool_name(call.get("tool_name", ""))
                for call in item.get("tool_calls", [])
                if isinstance(call, dict)
            ]
            for tool in memory_tools:
                tool = modern_tool_name(str(tool))
                if tool not in tools:
                    tools.append(tool)
        return tools

    def _heuristic_tools(self, state: WorkflowState) -> list[str]:
        return [
            "statistical_analysis_tool",
            "trend_analysis_tool",
            "changepoint_trend_tool",
        ]


class ActionModule:
    """Deterministic execution interface for the multi-view toolkit."""

    def __init__(self, toolkit: MultiViewToolkit | None = None) -> None:
        self.toolkit = toolkit or MultiViewToolkit()

    def execute(
        self, state: WorkflowState, config: CastFlowConfig
    ) -> list[ToolObservation]:
        state.baseline_forecast = None
        if hasattr(self.toolkit, "set_plan"):
            self.toolkit.set_plan(state.tool_schedule)
        observations = []
        allowed = set(PlanningModule.mandatory_tools + optional_tools(config))
        for name in state.tool_schedule:
            if name not in allowed:
                raise ValueError(f"unknown or disabled tool: {name}")
            try:
                observations.append(self.toolkit.execute(name, state.window, config))
            except (
                ArithmeticError,
                ValueError,
                RuntimeError,
                OSError,
                ImportError,
            ) as exc:
                observations.append(
                    ToolObservation(
                        name, "unavailable", {"error": f"{type(exc).__name__}: {exc}"}
                    )
                )
        aux = next(
            (
                obs.output
                for obs in observations
                if obs.tool_name == "model_auxiliary_tool"
            ),
            None,
        )
        if aux:
            candidate = aux.get("ensemble_forecast_baseline")
            if candidate is None:
                candidate = aux.get("reference_prediction")
            if structurally_valid(candidate, config.horizon):
                state.baseline_forecast = [float(value) for value in candidate]
        return observations


class ForecastingModule:
    """Specialized numerical forecaster backed by the configured API."""

    def __init__(self, llm: OpenAICompatibleLLM | None = None) -> None:
        self.llm = llm

    def forecast(self, state: WorkflowState, config: CastFlowConfig) -> list[float]:
        state.forecast_full_response = ""
        expected = state.expected_timestamps or forecast_timestamps(
            state.window, config.horizon
        )
        system, user = build_api_forecasting_messages(
            history=state.window.history,
            future=state.window.covariate_forecasts,
            target_col=state.window.target_col,
            timestamp_col=state.window.timestamp_col,
            horizon=config.horizon,
            lookback=config.lookback,
            tool_outputs=state.observation_map(),
            tool_schedule=state.tool_schedule,
            retry_feedback=state.reflection,
            retrieved_memory=state.retrieved_memory[: config.memory_top_k],
            expected_timestamps=expected,
        )
        state.forecast_full_prompt = user
        if not structurally_valid(state.baseline_forecast, config.horizon):
            state.reflection = append_note(
                state.reflection,
                "A usable ensemble prior is required for evidence-conditioned refinement.",
            )
            return invalid_forecast(config)
        if self.llm is None or not self.llm.available():
            state.reflection = append_note(
                state.reflection, "Forecasting model is unavailable."
            )
            return invalid_forecast(config)
        response = self.llm.chat(
            system=system,
            user=user,
            max_tokens=config.forecasting_max_tokens,
            temperature=config.forecasting_temperature,
        )
        if not response.ok:
            state.reflection = append_note(
                state.reflection, f"Forecasting model unavailable: {response.error}"
            )
            return invalid_forecast(config)
        reasoning = getattr(response, "reasoning_content", "") or ""
        state.forecast_full_response = (
            f"<think>{reasoning}</think>\n{response.content}"
            if reasoning and not re.search(r"<think\b", response.content, re.IGNORECASE)
            else response.content
        )
        if getattr(response, "finish_reason", None) in {"length", "content_filter"}:
            state.reflection = append_note(
                state.reflection, "Forecast response was truncated or filtered."
            )
            return invalid_forecast(config)
        parsed = parse_forecast_sequence(
            response.content, config.horizon, expected_timestamps=expected or None
        )
        if parsed is None:
            state.reflection = append_note(
                state.reflection,
                "Forecast response must contain exactly the required finite numbers aligned with the forecast positions.",
            )
            return invalid_forecast(config)
        return [float(value) for value in parsed]

    def _api_forecast(
        self, state: WorkflowState, config: CastFlowConfig
    ) -> list[float] | None:
        values = self.forecast(state, config)
        return values if structurally_valid(values, config.horizon) else None


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
        if not structurally_valid(state.forecast, config.horizon):
            return False, append_note(
                state.reflection,
                "FAIL: forecast is absent, non-finite, or has an incorrect horizon.",
            )
        if self.llm is None or not self.llm.available():
            return False, append_note(
                state.reflection, "FAIL: semantic reflection is unavailable."
            )
        message = self._api_reflect(state, config, metrics=metrics)
        verdict = reflection_verdict(message)
        if verdict is None:
            return False, append_note(
                message,
                "FAIL: reflection did not return an unambiguous PASS or FAIL verdict.",
            )
        return verdict, message

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
            "output_contract": "Return PASS or FAIL followed by one concise reason.",
        }
        if config.allow_train_only_tools and metrics:
            payload["training_trajectory_metrics"] = {
                key: float(value)
                for key, value in metrics.items()
                if np.isfinite(value)
            }
        response = self.llm.chat(
            system="You are the frozen CastFlow reflection module. Check format and evidence alignment.",
            user=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            max_tokens=config.reflection_max_tokens,
            temperature=0.0,
        )
        if not response.ok or getattr(response, "finish_reason", None) in {
            "length",
            "content_filter",
        }:
            return f"FAIL [reflection_api_unavailable: {response.error or 'response truncated or filtered'}]"
        return response.content.strip()

    def to_result(self, state: WorkflowState) -> ForecastResult:
        return ForecastResult(
            forecast=list(state.forecast) if state.forecast is not None else [],
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
        if isinstance(tools, list) and all(isinstance(tool, str) for tool in tools):
            return list(dict.fromkeys(modern_tool_name(tool) for tool in tools))
    except (ValueError, TypeError, AttributeError):
        pass
    return []


def extract_json_object(text: str) -> str:
    text = cleanup_jsonish(extract_answer_text(text))
    payload = json.loads(
        text,
        object_pairs_hook=_unique_planner_object,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite planner value")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("planning requires one complete JSON object")
    return text


def _unique_planner_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate planner JSON key")
        result[key] = value
    return result


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


def structurally_valid(values: object, horizon: int) -> bool:
    if not isinstance(values, (list, tuple, np.ndarray)):
        return False
    try:
        if len(values) != horizon or any(
            isinstance(value, (bool, np.bool_)) for value in values
        ):
            return False
        array = np.asarray(values, dtype=float)
        return array.shape == (horizon,) and bool(np.all(np.isfinite(array)))
    except (TypeError, ValueError, OverflowError):
        return False


def reflection_verdict(text: str) -> bool | None:
    text = extract_answer_text(text)
    words = {
        word.upper() for word in re.findall(r"\b(?:PASS|FAIL)\b", text, re.IGNORECASE)
    }
    if len(words) != 1:
        return None
    match = re.match(r"\s*(PASS|FAIL)\b", text, re.IGNORECASE)
    return match.group(1).upper() == "PASS" if match else None


def optional_tools(config: CastFlowConfig) -> list[str]:
    tools = [
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
        tools.append("autoregressive_residual_tool")
    return tools
