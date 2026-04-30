from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import TypeVar
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd

from .config import CastFlowConfig
from .data import load_dataset, make_latest_window, make_windows, write_forecast_csv
from .evaluation import extract_answer_text
from .llm import OpenAICompatibleLLM
from .memory import StrategyMemory
from .metrics import mae, mse
from .modules import ActionModule, ForecastingModule, PlanningModule, ReflectionModule, parse_tool_schedule
from .prompts import series_to_json
from .schema import ForecastResult, TimeSeriesWindow, WorkflowState

T = TypeVar("T")


class CastFlow:
    """Planning-action-forecasting-reflection workflow."""

    def __init__(
        self,
        config: CastFlowConfig | None = None,
        memory: StrategyMemory | None = None,
    ) -> None:
        self.config = config or CastFlowConfig.from_env()
        if config is not None:
            env_config = CastFlowConfig.from_env()
            if self.config.api_base_url is None:
                self.config.api_base_url = env_config.api_base_url
            if self.config.api_key is None:
                self.config.api_key = env_config.api_key
            if self.config.api_model is None:
                self.config.api_model = env_config.api_model
        self.config.validate()
        self.memory = memory or StrategyMemory()
        self.llm = OpenAICompatibleLLM(self.config)
        self.local_llm = OpenAICompatibleLLM(
            self.config,
            base_url=self.config.local_model_base_url,
            api_key=self.config.local_model_api_key,
            model=self.config.local_model_name,
            enabled=self.config.local_forecast_ready(),
        )
        self.planning = PlanningModule(self.llm)
        self.action = ActionModule()
        self.forecasting = ForecastingModule(self.llm)
        self.local_forecasting = ForecastingModule(self.local_llm) if self.local_llm.available() else self.forecasting
        self.reflection = ReflectionModule(self.llm)

    def run_window(
        self,
        window: TimeSeriesWindow,
        forecasting: ForecastingModule | None = None,
    ) -> ForecastResult:
        matches = self.memory.retrieve(
            window.target_history,
            top_k=self.config.memory_top_k,
            similarity_threshold=self.config.memory_similarity_threshold,
        )
        state = WorkflowState(window=window, retrieved_memory=[match.entry for match in matches])
        forecasting_module = forecasting or self.forecasting

        while state.retry_count < self.config.test_max_retry_loops:
            state.tool_schedule = self.planning.plan(state, self.config)
            state.observations = self.action.execute(state, self.config)
            state.forecast = forecasting_module.forecast(state, self.config)
            valid, message = self.reflection.validate(state, self.config)
            state.reflection = message
            if valid:
                break
            state.retry_count += 1

        result = self.reflection.to_result(state)
        if window.target_future is not None:
            result.metrics.update(
                {
                    "mse": mse(result.forecast, window.target_future),
                    "mae": mae(result.forecast, window.target_future),
                }
            )
        return result

    def run_memory_window(
        self,
        window: TimeSeriesWindow,
        window_idx: int | None = None,
        verbose: bool = False,
    ) -> ForecastResult:
        """Build-memory path aligned with the old reflector loop.

        The old implementation kept retrying a training sample until the forecast
        was excellent, then saved that strategy. If no excellent strategy appeared,
        it saved the best format/LLM-passed attempt after the reflection limit.
        """
        matches = self.memory.retrieve(
            window.target_history,
            top_k=self.config.memory_top_k,
            similarity_threshold=self.config.memory_similarity_threshold,
        )
        log_sample_progress(
            verbose,
            f"[sample {display_idx(window_idx)}] start; retrieved_memory={len(matches)}; max_attempts={self.config.max_retry_loops}",
        )
        state = WorkflowState(window=window, retrieved_memory=[match.entry for match in matches])
        best_result: ForecastResult | None = None
        best_mse = float("inf")
        last_result: ForecastResult | None = None
        attempts = max(1, self.config.max_retry_loops)

        for attempt in range(attempts):
            state.retry_count = attempt
            log_sample_progress(verbose, f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] planning...")
            base_schedule = self.planning.plan(state, self.config)
            log_sample_progress(
                verbose,
                f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] base_tools={base_schedule}",
            )

            result, valid = self.run_parallel_memory_candidates(
                state,
                base_schedule=base_schedule,
                attempt=attempt,
                attempts=attempts,
                window_idx=window_idx,
                verbose=verbose,
            )
            last_result = result

            if valid:
                current_mse = result.metrics.get("mse", float("inf"))
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_result = result
                    log_sample_progress(
                        verbose,
                        (
                            f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] "
                            f"best_so_far mse={format_metric(result.metrics.get('mse'))} "
                            f"mae={format_metric(result.metrics.get('mae'))}"
                        ),
                    )
                if is_excellent_memory_result(result, self.config):
                    result.metrics["train_excellent"] = 1.0
                    result.metrics["memory_save"] = 1.0
                    log_sample_progress(
                        verbose,
                        (
                            f"[sample {display_idx(window_idx)}] excellent; "
                            f"mse={format_metric(result.metrics.get('mse'))} "
                            f"mae={format_metric(result.metrics.get('mae'))}; selected for memory"
                        ),
                    )
                    return result

            state.retry_count += 1
            state.reflection = result.reflection

        selected = best_result or last_result
        if selected is None:
            raise ValueError("memory window produced no forecast result")
        selected.metrics["train_excellent"] = 1.0 if is_excellent_memory_result(selected, self.config) else 0.0
        selected.metrics["memory_save"] = 1.0 if best_result is not None else 0.0
        if selected.metrics["memory_save"] >= 1.0:
            log_sample_progress(
                verbose,
                (
                    f"[sample {display_idx(window_idx)}] no excellent result; "
                    f"saving best PASS attempt mse={format_metric(selected.metrics.get('mse'))} "
                    f"mae={format_metric(selected.metrics.get('mae'))}"
                ),
            )
        else:
            log_sample_progress(verbose, f"[sample {display_idx(window_idx)}] no valid PASS attempt; skipped")
        return selected

    def run_parallel_memory_candidates(
        self,
        state: WorkflowState,
        base_schedule: list[str],
        attempt: int,
        attempts: int,
        window_idx: int | None,
        verbose: bool,
    ) -> tuple[ForecastResult, bool]:
        schedules = self.parallel_tool_schedules(state, base_schedule)
        log_sample_progress(
            verbose,
            f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] K-parallel candidates={len(schedules)}",
        )
        results: list[tuple[int, ForecastResult, bool]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(schedules))) as pool:
            future_to_idx = {
                pool.submit(self.run_memory_candidate, state, schedule): idx
                for idx, schedule in enumerate(schedules)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, valid = future.result()
                    results.append((idx, result, valid))
                    log_sample_progress(
                        verbose,
                        (
                            f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] "
                            f"candidate {idx + 1}/{len(schedules)} {'PASS' if valid else 'FAIL'} "
                            f"mse={format_metric(result.metrics.get('mse'))} "
                            f"mae={format_metric(result.metrics.get('mae'))} "
                            f"tools={result.tool_schedule}"
                        ),
                    )
                except Exception as exc:
                    log_sample_progress(
                        verbose,
                        (
                            f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] "
                            f"candidate {idx + 1}/{len(schedules)} ERROR {type(exc).__name__}: {exc}"
                        ),
                    )

        if not results:
            raise ValueError("all K-parallel candidates failed before producing a result")
        valid_results = [item for item in results if item[2]]
        selected_idx, selected_result, selected_valid = min(valid_results or results, key=lambda item: metric_for_selection(item[1]))
        log_sample_progress(
            verbose,
            (
                f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] "
                f"selected candidate {selected_idx + 1}/{len(schedules)} "
                f"{'PASS' if selected_valid else 'FAIL'} "
                f"mse={format_metric(selected_result.metrics.get('mse'))} "
                f"mae={format_metric(selected_result.metrics.get('mae'))}"
            ),
        )
        return selected_result, selected_valid

    def run_memory_candidate(
        self,
        base_state: WorkflowState,
        tool_schedule: list[str],
    ) -> tuple[ForecastResult, bool]:
        state = WorkflowState(
            window=base_state.window,
            retrieved_memory=base_state.retrieved_memory,
            tool_schedule=tool_schedule,
            reflection=base_state.reflection,
            retry_count=base_state.retry_count,
        )
        state.observations = self.action.execute(state, self.config)
        state.forecast = self.forecasting.forecast(state, self.config)
        current_metrics: dict[str, float] = {}
        if state.window.target_future is not None:
            current_metrics = {
                "mse": mse(state.forecast, state.window.target_future),
                "mae": mae(state.forecast, state.window.target_future),
            }
        valid, message = self.reflection.validate(state, self.config, metrics=current_metrics)
        state.reflection = message
        result = self.reflection.to_result(state)
        result.metrics.update(current_metrics)
        return result, valid

    def parallel_tool_schedules(self, state: WorkflowState, base_schedule: list[str]) -> list[list[str]]:
        k = max(1, self.config.parallel_plan_k)
        mandatory = PlanningModule.mandatory_tools
        allowed_optional = optional_tool_names(self.config)
        baseline_optional = [tool for tool in base_schedule if tool in allowed_optional]
        strategies = [baseline_optional or ["statistical_analysis_tool", "trend_analysis_tool", "changepoint_trend_tool"]]
        strategies.extend(self.api_parallel_optional_strategies(state, strategies[0], k - 1))
        strategies.extend(deterministic_parallel_optional_strategies(state, self.config))

        schedules: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for strategy in strategies:
            schedule = list(mandatory)
            schedule.extend(tool for tool in strategy if tool in allowed_optional and tool not in schedule)
            key = tuple(schedule)
            if key in seen:
                continue
            seen.add(key)
            schedules.append(schedule)
            if len(schedules) >= k:
                break
        return schedules or [list(mandatory)]

    def api_parallel_optional_strategies(
        self,
        state: WorkflowState,
        baseline_optional: list[str],
        needed: int,
    ) -> list[list[str]]:
        if needed <= 0 or self.llm is None or not self.llm.available():
            return []
        allowed_optional = optional_tool_names(self.config)
        payload = {
            "task": "Generate diverse alternative optional tool strategies for build-memory K-parallel forecasting.",
            "mandatory_tools": PlanningModule.mandatory_tools,
            "optional_tools": allowed_optional,
            "baseline_strategy": baseline_optional,
            "num_additional_strategies": needed,
            "target_tail": state.window.target_history[-min(24, len(state.window.target_history)) :].round(6).tolist(),
            "output_contract": {"strategies": [["optional_tool_1", "optional_tool_2"]]},
        }
        response = self.llm.chat(
            system="You are a CastFlow planning expert. Return only JSON with diverse optional tool strategies.",
            user=json.dumps(payload, ensure_ascii=False),
            max_tokens=self.config.planner_max_tokens,
            temperature=0.3,
        )
        if not response.ok:
            return []
        try:
            raw_strategies = json.loads(response.content).get("strategies", [])
        except Exception:
            raw_strategies = [parse_tool_schedule(response.content)]

        strategies: list[list[str]] = []
        for raw in raw_strategies:
            if isinstance(raw, list):
                strategy = [str(tool) for tool in raw if str(tool) in allowed_optional]
            else:
                strategy = [tool for tool in parse_tool_schedule(str(raw)) if tool in allowed_optional]
            if strategy:
                strategies.append(strategy)
        return strategies[:needed]

    def build_memory_from_csv(
        self,
        data_path: str | Path,
        output_path: str | Path,
        stride: int | None = None,
        max_windows: int | None = None,
        show_progress: bool = False,
        resume: bool = False,
        verbose_samples: bool = False,
    ) -> StrategyMemory:
        previous_train_only = self.config.allow_train_only_tools
        self.config.allow_train_only_tools = True
        try:
            dataset = load_dataset(data_path, self.config.target_col, self.config.timestamp_col)
            windows = make_windows(
                dataset,
                lookback=self.config.lookback,
                horizon=self.config.horizon,
                stride=stride,
                max_windows=max_windows,
            )
            if not windows:
                raise ValueError("no windows could be built; check lookback/horizon against dataset length")

            output = Path(output_path)
            if resume and output.exists():
                self.memory = StrategyMemory.load(output)
            completed = completed_window_indices(self.memory, self.config.dataset_name)

            for idx, window in enumerate(progress_iter(windows, label="Building strategy memory", unit="window", enabled=show_progress)):
                if resume and idx in completed:
                    log_sample_progress(verbose_samples, f"[sample {idx}] already exists in memory; skipped by resume")
                    continue
                result = self.run_memory_window(window, window_idx=idx, verbose=verbose_samples)
                if result.metrics.get("memory_save", 1.0) < 1.0:
                    log_sample_progress(verbose_samples, f"[sample {idx}] not written; memory_entries={len(self.memory.entries)}")
                    continue
                aux = result.observations.get("model_auxiliary_tool", {})
                self.memory.add(
                    target_history=window.target_history,
                    tool_schedule=result.tool_schedule,
                    diagnostic_outputs=result.observations,
                    trajectory=result.reflection,
                    metrics={"window_idx": float(idx), **result.metrics},
                    input_json=series_to_json(window.history, window.target_col, window.timestamp_col),
                    ground_truth=series_to_json(window.future, window.target_col, window.timestamp_col) if window.future is not None else "",
                    forecast_full_prompt=result.forecast_full_prompt,
                    forecast_answer=result.forecast_full_response,
                    model_name=",".join(sorted(aux.get("best_model", {}).keys())) if isinstance(aux.get("best_model"), dict) else "",
                    dataset_name=self.config.dataset_name,
                )
                self.memory.save(output_path)
                log_sample_progress(verbose_samples, f"[sample {idx}] written immediately -> {output_path}; memory_entries={len(self.memory.entries)}")
            self.memory.save(output_path)
            return self.memory
        finally:
            self.config.allow_train_only_tools = previous_train_only

    def forecast_csv(
        self,
        data_path: str | Path,
        output_path: str | Path,
        stride: int | None = None,
        max_windows: int | None = None,
        latest: bool = False,
        show_progress: bool = False,
    ) -> ForecastResult | list[ForecastResult]:
        dataset = load_dataset(data_path, self.config.target_col, self.config.timestamp_col)
        forecasting = self.local_forecasting
        windows = [] if latest else make_windows(
            dataset,
            lookback=self.config.lookback,
            horizon=self.config.horizon,
            stride=stride,
            max_windows=max_windows,
        )
        if windows:
            results: list[ForecastResult] = []
            rows: list[dict[str, object]] = []
            for idx, window in enumerate(progress_iter(windows, label="Forecasting test windows", unit="window", enabled=show_progress)):
                result = self.run_window(window, forecasting=forecasting)
                results.append(result)
                aux = result.observations.get("model_auxiliary_tool", {})
                full_answer = result.forecast_full_response
                rows.append(
                    {
                        "idx": idx,
                        "model_name": ",".join(sorted(aux.get("best_model", {}).keys())) if isinstance(aux.get("best_model"), dict) else "",
                        "answer": extract_answer_text(full_answer),
                        "ground_truth": series_to_json(window.future, window.target_col, window.timestamp_col) if window.future is not None else "",
                        "prompt": result.forecast_full_prompt,
                        "full_answer": full_answer,
                        "mse": result.metrics.get("mse", ""),
                        "mae": result.metrics.get("mae", ""),
                        "tools": json.dumps(result.tool_schedule, ensure_ascii=False),
                        "reflection": result.reflection,
                    }
                )
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(output_path, index=False)
            return results

        window = make_latest_window(dataset, self.config.lookback)
        result = self.run_window(window, forecasting=forecasting)

        start_ts = None
        freq = None
        if dataset.timestamp_col:
            timestamps = pd.to_datetime(dataset.frame[dataset.timestamp_col], errors="coerce")
            freq = pd.infer_freq(timestamps.dropna())
            if freq:
                start_ts = timestamps.iloc[-1] + pd.tseries.frequencies.to_offset(freq)
        write_forecast_csv(output_path, result.forecast, start_timestamp=start_ts, frequency=freq)
        return result


def progress_iter(items: Sequence[T], *, label: str, unit: str, enabled: bool) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    try:
        from tqdm.auto import tqdm

        yield from tqdm(items, total=len(items), desc=label, unit=unit)
        return
    except Exception:
        yield from items


def completed_window_indices(memory: StrategyMemory, dataset_name: str | None = None) -> set[int]:
    completed: set[int] = set()
    for entry in memory.entries:
        entry_dataset = entry.get("dataset_name")
        if dataset_name is not None and entry_dataset not in (dataset_name, None, ""):
            continue
        metrics = entry.get("metrics", {})
        value = metrics.get("window_idx", entry.get("idx"))
        try:
            completed.add(int(value))
        except (TypeError, ValueError):
            continue
    return completed


def is_excellent_memory_result(result: ForecastResult, config: CastFlowConfig) -> bool:
    result_mse = result.metrics.get("mse")
    result_mae = result.metrics.get("mae")
    if result_mse is None or result_mae is None:
        return False
    return result_mse < config.excellent_mse_threshold and result_mae < config.excellent_mae_threshold


def optional_tool_names(config: CastFlowConfig) -> list[str]:
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


def deterministic_parallel_optional_strategies(
    state: WorkflowState,
    config: CastFlowConfig,
) -> list[list[str]]:
    strategies = [
        ["statistical_analysis_tool", "trend_analysis_tool", "changepoint_trend_tool"],
        ["basic_statistics_tool", "changepoint_trend_tool", "data_quality_tool", "event_summary_tool"],
    ]
    if len(state.window.history.columns) > 2:
        strategies.append(["cross_channel_tool", "event_summary_tool", "comprehensive_feature_tool"])
    if config.allow_train_only_tools:
        strategies.append(["autoregressive_residual_tool", "statistical_analysis_tool", "trend_analysis_tool"])
    return strategies


def metric_for_selection(result: ForecastResult) -> float:
    value = result.metrics.get("mse")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if score != score:
        return float("inf")
    return score


def log_sample_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def display_idx(idx: int | None) -> str:
    return "?" if idx is None else str(idx)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def compact_message(message: str, limit: int = 180) -> str:
    text = " ".join(str(message).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
