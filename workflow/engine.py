from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
from contextlib import nullcontext
import copy
import hashlib
import math
from collections.abc import Iterator, Sequence
from typing import TypeVar
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain, combinations
import json

import pandas as pd

from utils.config import CastFlowConfig
from utils.io import (
    load_dataset,
    make_latest_window,
    make_windows,
    write_forecast_csv,
    forecast_timestamps,
    atomic_write_csv,
    attach_covariate_forecasts,
)
from utils.llm import OpenAICompatibleLLM
from utils.prompting import prompt_identity, render_prompt
from workflow.memory import StrategyMemory
from workflow.recovery import ExecutionJournal, digest
from utils.schema import ForecastResult, TimeSeriesWindow, WorkflowState
from forecasting.modules import (
    ActionModule,
    ForecastingModule,
    PlanningModule,
    ReflectionModule,
    parse_tool_schedule,
    structurally_valid,
    invalid_forecast,
    optional_tools,
)
from forecasting.prompts import series_to_json
from utils.metrics import mae, mse

T = TypeVar("T")


def bounded_memory_windows(
    dataset,
    lookback,
    horizon,
    *,
    stride=None,
    max_windows=None,
    start_origin_row=None,
    end_row=None,
):
    """Return source origins and windows on the original chronological stride grid."""
    for name, value in (("lookback", lookback), ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    step = horizon if stride is None else stride
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("memory stride must be a positive integer")
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows < 1
    ):
        raise ValueError("max_windows must be a positive integer or None")
    start = lookback if start_origin_row is None else start_origin_row
    stop = len(dataset.frame) if end_row is None else end_row
    for name, value in (("start_origin_row", start), ("end_row", stop)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer row boundary")
    if not lookback <= start < stop <= len(dataset.frame):
        raise ValueError("memory target boundaries must stay within the training CSV")
    first = lookback + max(0, (start - lookback + step - 1) // step) * step
    windows = []
    for origin in range(first, stop - horizon + 1, step):
        windows.append(
            (
                origin,
                TimeSeriesWindow(
                    history=dataset.frame.iloc[origin - lookback : origin].copy(),
                    future=dataset.frame.iloc[origin : origin + horizon].copy(),
                    target_col=dataset.target_col,
                    timestamp_col=dataset.timestamp_col,
                ),
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


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
        self.local_forecasting = ForecastingModule(self.local_llm)
        self.reflection = ReflectionModule(self.llm)

    def run_window(
        self,
        window: TimeSeriesWindow,
        forecasting: ForecastingModule | None = None,
        *,
        request_id: str | None = None,
    ) -> ForecastResult:
        """Forecast with target-free decisions and explicit terminal fallbacks."""
        if self.config.allow_train_only_tools:
            worker = copy.copy(self)
            worker.config = replace(self.config, allow_train_only_tools=False)
            return worker.run_window(window, forecasting, request_id=request_id)
        expected = forecast_timestamps(window, self.config.horizon)
        observed = replace(
            window,
            future=None,
            history=window.history.copy(deep=True),
            covariate_forecasts=(
                window.covariate_forecasts.copy(deep=True)
                if window.covariate_forecasts is not None
                else None
            ),
        )
        if observed.covariate_forecasts is not None:
            attach_covariate_forecasts(
                observed,
                observed.covariate_forecasts,
                self.config.horizon,
                issued_at=observed.covariate_issued_at,
            )
        matches = self.memory.retrieve(
            observed.target_history,
            top_k=self.config.memory_top_k,
            similarity_threshold=self.config.memory_similarity_threshold,
        )
        state = WorkflowState(
            window=observed,
            retrieved_memory=[match.entry for match in matches],
            expected_timestamps=expected,
        )
        model = forecasting or self.local_forecasting
        valid_history: list[ForecastResult] = []
        latest_prior: ForecastResult | None = None
        journal = None
        if self.config.execution_journal_dir:
            identifier = request_id or self.config.execution_request_id
            if not identifier:
                raise ValueError("stage recovery requires an explicit request_id")
            identity = request_identity(observed, self.config, model)
            identity["expected_timestamps"] = expected
            journal = ExecutionJournal(
                self.config.execution_journal_dir, identifier, identity
            )
        toolkit = self.action.toolkit
        scope = (
            toolkit.request_scope(observed, self.config)
            if hasattr(toolkit, "request_scope")
            else nullcontext()
        )
        with journal if journal is not None else nullcontext(), scope:
            for attempt in range(self.config.test_max_retry_loops + 1):
                state.retry_count = attempt
                run = (
                    journal.run
                    if journal is not None
                    else lambda name, inputs, fn: fn()
                )
                prefix = f"revision:{attempt}"
                plan = run(
                    prefix + ":plan",
                    {
                        "feedback": state.reflection,
                        "history": state.history,
                        "memory": state.retrieved_memory,
                    },
                    lambda: self._generate_plan(state),
                )
                state.tool_schedule = plan["schedule"]
                state.reflection = plan["feedback"]
                action = run(
                    prefix + ":action",
                    {"schedule": state.tool_schedule},
                    lambda: self._execute_action(state),
                )
                from utils.schema import ToolObservation

                state.observations = [
                    ToolObservation(**row) for row in action["observations"]
                ]
                state.baseline_forecast = action["baseline"]
                if structurally_valid(state.baseline_forecast, self.config.horizon):
                    latest_prior = copy.deepcopy(self.reflection.to_result(state))
                forecast = run(
                    prefix + ":forecast",
                    {
                        "observations": action,
                        "feedback": state.reflection,
                        "memory": state.retrieved_memory,
                    },
                    lambda: self._generate_forecast(state, model),
                )
                state.forecast = (
                    forecast["forecast"]
                    if forecast["forecast"] is not None
                    else invalid_forecast(self.config)
                )
                state.forecast_full_prompt = forecast["prompt"]
                state.forecast_system_prompt = forecast.get("system_prompt", "")
                state.forecast_full_response = forecast["response"]
                state.reflection = forecast["feedback"]
                if structurally_valid(state.forecast, self.config.horizon):
                    valid_history.append(
                        copy.deepcopy(self.reflection.to_result(state))
                    )
                valid, message = run(
                    prefix + ":validate",
                    {"forecast": forecast, "observations": action},
                    lambda: list(self.reflection.validate(state, self.config)),
                )
                state.reflection = message
                if valid:
                    result = self.reflection.to_result(state)
                    result.status = "accepted"
                    break
                state.history.append(
                    {
                        "attempt": attempt + 1,
                        "tools": list(state.tool_schedule),
                        "observations": state.observation_map(),
                        "response": state.forecast_full_response,
                        "feedback": message,
                    }
                )
            else:
                if valid_history:
                    result = valid_history[-1]
                    result.status = "response_fallback"
                    result.fallback_source = "request_history"
                elif latest_prior is not None:
                    result = latest_prior
                    result.forecast = list(latest_prior.baseline_forecast)
                    result.forecast_full_response = serialized_forecast(
                        result.forecast, expected
                    )
                    result.status = "prior_fallback"
                    result.fallback_source = "ensemble_prior"
                else:
                    result = self.reflection.to_result(state)
                    result.forecast = []
                    result.status = "missing"
                    result.forecast_full_response = ""
                result.failure_reason = state.reflection
                result.reflection = (
                    "Fallback after bounded verification: " + state.reflection
                )
        result.attempts = attempt + 1
        if window.target_future is not None and structurally_valid(
            result.forecast, self.config.horizon
        ):
            result.metrics.update(
                {
                    "mse": mse(result.forecast, window.target_future),
                    "mae": mae(result.forecast, window.target_future),
                }
            )
        return result

    def _execute_action(self, state: WorkflowState) -> dict:
        observations = self.action.execute(state, self.config)
        return {
            "observations": [asdict(item) for item in observations],
            "baseline": state.baseline_forecast,
        }

    def _generate_plan(self, state: WorkflowState) -> dict:
        schedule = self.planning.plan(state, self.config)
        return {"schedule": schedule, "feedback": state.reflection}

    def _generate_forecast(
        self, state: WorkflowState, module: ForecastingModule
    ) -> dict:
        values = module.forecast(state, self.config)
        return {
            "forecast": (
                values if structurally_valid(values, self.config.horizon) else None
            ),
            "prompt": state.forecast_full_prompt,
            "system_prompt": state.forecast_system_prompt,
            "response": state.forecast_full_response,
            "feedback": state.reflection,
        }

    def run_memory_window(
        self,
        window: TimeSeriesWindow,
        window_idx: int | None = None,
        verbose: bool = False,
    ) -> ForecastResult:
        """Retain the lowest-error trajectory among all valid explored paths."""
        matches = self.memory.retrieve(
            window.target_history,
            top_k=self.config.memory_top_k,
            similarity_threshold=self.config.memory_similarity_threshold,
        )
        log_sample_progress(
            verbose,
            f"[sample {display_idx(window_idx)}] start; retrieved_memory={len(matches)}; max_attempts={self.config.max_retry_loops}",
        )
        state = WorkflowState(
            window=window, retrieved_memory=[match.entry for match in matches]
        )
        best_result: ForecastResult | None = None
        best_mse = float("inf")
        last_result: ForecastResult | None = None
        attempts = self.config.max_retry_loops + 1

        for attempt in range(attempts):
            state.retry_count = attempt
            log_sample_progress(
                verbose,
                f"[sample {display_idx(window_idx)}][attempt {attempt + 1}/{attempts}] planning...",
            )
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
            state.retry_count += 1
            state.reflection = result.reflection

        selected = best_result or last_result
        if selected is None:
            raise ValueError("memory window produced no forecast result")
        selected.metrics["train_excellent"] = (
            1.0 if is_excellent_memory_result(selected, self.config) else 0.0
        )
        selected.metrics["memory_save"] = 1.0 if best_result is not None else 0.0
        if selected.metrics["memory_save"] >= 1.0:
            log_sample_progress(
                verbose,
                (
                    f"[sample {display_idx(window_idx)}] saving best PASS attempt "
                    f"mse={format_metric(selected.metrics.get('mse'))} "
                    f"mae={format_metric(selected.metrics.get('mae'))}"
                ),
            )
        else:
            log_sample_progress(
                verbose,
                f"[sample {display_idx(window_idx)}] no valid PASS attempt; skipped",
            )
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
            raise ValueError(
                "all K-parallel candidates failed before producing a result"
            )
        valid_results = [item for item in results if item[2]]
        selected_idx, selected_result, selected_valid = min(
            valid_results or results,
            key=lambda item: (metric_for_selection(item[1]), item[0]),
        )
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
            window=replace(
                base_state.window,
                future=None,
                history=base_state.window.history.copy(deep=True),
                covariate_forecasts=(
                    base_state.window.covariate_forecasts.copy(deep=True)
                    if base_state.window.covariate_forecasts is not None
                    else None
                ),
            ),
            expected_timestamps=forecast_timestamps(
                base_state.window, self.config.horizon
            ),
            retrieved_memory=base_state.retrieved_memory,
            tool_schedule=tool_schedule,
            reflection=base_state.reflection,
            retry_count=base_state.retry_count,
        )
        if state.window.covariate_forecasts is not None:
            attach_covariate_forecasts(
                state.window,
                state.window.covariate_forecasts,
                self.config.horizon,
                issued_at=state.window.covariate_issued_at,
            )
        toolkit = self.action.toolkit
        scope = (
            toolkit.request_scope(state.window, self.config)
            if hasattr(toolkit, "request_scope")
            else nullcontext()
        )
        with scope:
            state.observations = self.action.execute(state, self.config)
            state.forecast = self.forecasting.forecast(state, self.config)
        current_metrics: dict[str, float] = {}
        if base_state.window.target_future is not None and structurally_valid(
            state.forecast, self.config.horizon
        ):
            current_metrics = {
                "mse": mse(state.forecast, base_state.window.target_future),
                "mae": mae(state.forecast, base_state.window.target_future),
            }
        valid, message = self.reflection.validate(
            state, self.config, metrics=current_metrics
        )
        state.reflection = message
        result = self.reflection.to_result(state)
        result.metrics.update(current_metrics)
        return result, valid and math.isfinite(metric_for_selection(result))

    def parallel_tool_schedules(
        self, state: WorkflowState, base_schedule: list[str]
    ) -> list[list[str]]:
        k = max(1, self.config.parallel_plan_k)
        mandatory = PlanningModule.mandatory_tools
        allowed_optional = optional_tool_names(self.config)
        if k > 2 ** len(allowed_optional):
            raise ValueError(
                "parallel_plan_k exceeds the number of available optional tool subsets"
            )
        baseline_optional = [tool for tool in base_schedule if tool in allowed_optional]
        strategies = [baseline_optional]
        strategies.extend(
            self.api_parallel_optional_strategies(state, strategies[0], k - 1)
        )
        strategies.extend(
            deterministic_parallel_optional_strategies(state, self.config)
        )

        schedules: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        remaining_subsets = (
            list(subset)
            for size in range(len(allowed_optional) + 1)
            for subset in combinations(allowed_optional, size)
        )
        for strategy in chain(strategies, remaining_subsets):
            schedule = list(mandatory)
            schedule.extend(
                tool
                for tool in strategy
                if tool in allowed_optional and tool not in schedule
            )
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
            "role": "exploration",
            "mandatory_tools": PlanningModule.mandatory_tools,
            "optional_tools": allowed_optional,
            "baseline_strategy": baseline_optional,
            "num_additional_strategies": needed,
            "target_tail": state.window.target_history[
                -min(24, len(state.window.target_history)) :
            ]
            .round(6)
            .tolist(),
            "output_contract": {"strategies": [["optional_tool_1", "optional_tool_2"]]},
        }
        response = self.llm.chat(
            system=render_prompt(self.config, "exploration_system"),
            user=render_prompt(
                self.config,
                "exploration_user",
                context=json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, sort_keys=True
                ),
            ),
            max_tokens=self.config.planner_max_tokens,
            temperature=0.3,
        )
        if not response.ok or getattr(response, "finish_reason", None) in {
            "length",
            "content_filter",
        }:
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
                strategy = [
                    tool
                    for tool in parse_tool_schedule(str(raw))
                    if tool in allowed_optional
                ]
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
        covariate_forecasts_path: str | Path | None = None,
        covariate_issued_at: str | None = None,
        covariate_issue_col: str = "issued_at",
        completion_journal_dir: str | Path | None = None,
        start_origin_row: int | None = None,
        end_row: int | None = None,
    ) -> StrategyMemory:
        worker = copy.copy(self)
        worker.config = replace(self.config, allow_train_only_tools=True)
        memory = worker._build_memory_from_csv(
            data_path,
            output_path,
            stride,
            max_windows,
            show_progress,
            resume,
            verbose_samples,
            covariate_forecasts_path,
            covariate_issued_at,
            covariate_issue_col,
            completion_journal_dir,
            start_origin_row,
            end_row,
        )
        self.memory = memory
        return memory

    def _build_memory_from_csv(
        self,
        data_path: str | Path,
        output_path: str | Path,
        stride: int | None = None,
        max_windows: int | None = None,
        show_progress: bool = False,
        resume: bool = False,
        verbose_samples: bool = False,
        covariate_forecasts_path: str | Path | None = None,
        covariate_issued_at: str | None = None,
        covariate_issue_col: str = "issued_at",
        completion_journal_dir: str | Path | None = None,
        start_origin_row: int | None = None,
        end_row: int | None = None,
    ) -> StrategyMemory:
        previous_train_only = self.config.allow_train_only_tools
        self.config.allow_train_only_tools = True
        try:
            source_fingerprint = hashlib.sha256(
                Path(data_path).read_bytes()
            ).hexdigest()
            dataset = load_dataset(
                data_path, self.config.target_col, self.config.timestamp_col
            )
            if (
                hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
                != source_fingerprint
            ):
                raise ValueError("training CSV changed while loading the memory inputs")
            source_windows = bounded_memory_windows(
                dataset,
                lookback=self.config.lookback,
                horizon=self.config.horizon,
                stride=stride,
                max_windows=max_windows,
                start_origin_row=start_origin_row,
                end_row=end_row,
            )
            if not source_windows:
                raise ValueError(
                    "no windows could be built; check lookback/horizon against dataset length"
                )
            source_origins, windows = zip(*source_windows)

            output = Path(output_path)
            if resume and output.exists():
                self.memory = StrategyMemory.load(output)
            completed = completed_window_indices(self.memory, self.config.dataset_name)
            covariate_fingerprint = (
                hashlib.sha256(Path(covariate_forecasts_path).read_bytes()).hexdigest()
                if covariate_forecasts_path
                else None
            )
            covariates = (
                pd.read_csv(covariate_forecasts_path, keep_default_na=False)
                if covariate_forecasts_path
                else None
            )
            if (
                covariate_forecasts_path
                and hashlib.sha256(
                    Path(covariate_forecasts_path).read_bytes()
                ).hexdigest()
                != covariate_fingerprint
            ):
                raise ValueError(
                    "covariate forecasts changed while loading the memory inputs"
                )
            build_identity = request_identity(windows[0], self.config, self.forecasting)
            build_identity.update(
                {
                    "source_fingerprint": source_fingerprint,
                    "covariate_source_fingerprint": covariate_fingerprint,
                    "covariate_issued_at": covariate_issued_at,
                    "covariate_issue_col": covariate_issue_col,
                    "stride": stride,
                    "start_origin_row": start_origin_row,
                    "end_row": end_row,
                    "source_origins": list(source_origins),
                }
            )
            build_fingerprint = digest(build_identity)
            if resume and any(
                entry.get("dataset_name", "") == (self.config.dataset_name or "")
                and entry.get("build_fingerprint") != build_fingerprint
                for entry in self.memory.entries
            ):
                raise ValueError(
                    "memory resume inputs, model, configuration or implementation changed; use a new output path"
                )

            journal = (
                ExecutionJournal(
                    completion_journal_dir,
                    "training-memory-windows",
                    {"build_fingerprint": build_fingerprint},
                )
                if completion_journal_dir
                else None
            )
            with journal if journal is not None else nullcontext():
                for idx, window in enumerate(
                    progress_iter(
                        windows,
                        label="Building strategy memory",
                        unit="window",
                        enabled=show_progress,
                    )
                ):
                    if covariates is not None:
                        attach_covariate_forecasts(
                            window,
                            covariates,
                            self.config.horizon,
                            issued_at=covariate_issued_at,
                            issue_col=covariate_issue_col,
                        )
                    input_json = series_to_json(
                        window.history, window.target_col, window.timestamp_col
                    )
                    matching = self._matching_memory_entry(
                        idx, input_json, source_fingerprint, build_fingerprint
                    )
                    if resume and idx in completed and matching is not None:
                        # Commit the accepted artifact before its completion marker.
                        self.memory.save(output_path)
                        complete = lambda: {
                            "retained": True,
                            "entry_hash": digest(matching),
                            "window_idx": idx,
                        }
                    else:
                        complete = lambda: self._retain_memory_window(
                            window,
                            idx,
                            output_path,
                            source_fingerprint,
                            build_fingerprint,
                            verbose_samples,
                            source_origins[idx],
                        )
                    if journal is not None:
                        outcome = journal.run(
                            f"window:{idx}",
                            {
                                "input_json": input_json,
                                "window_idx": idx,
                                "source_origin_row": source_origins[idx],
                            },
                            complete,
                        )
                        entry = self._matching_memory_entry(
                            idx, input_json, source_fingerprint, build_fingerprint
                        )
                        if not isinstance(outcome, dict) or not isinstance(
                            outcome.get("retained"), bool
                        ):
                            raise ValueError("invalid memory-window completion record")
                        if outcome["retained"]:
                            if entry is None or digest(entry) != outcome.get(
                                "entry_hash"
                            ):
                                raise ValueError(
                                    "completed accepted memory trajectory is missing or changed"
                                )
                        elif entry is not None:
                            raise ValueError(
                                "rejected memory-window completion conflicts with a retained trajectory"
                            )
                    else:
                        complete()
            self.memory.save(output_path)
            return self.memory
        finally:
            self.config.allow_train_only_tools = previous_train_only

    def _matching_memory_entry(
        self, idx, input_json, source_fingerprint, build_fingerprint
    ):
        matches = [
            entry
            for entry in self.memory.entries
            if entry.get("build_fingerprint") == build_fingerprint
            and entry.get("source_fingerprint") == source_fingerprint
            and entry.get("input_json") == input_json
            and entry.get("idx") == idx
            and entry.get("dataset_name", "") == (self.config.dataset_name or "")
        ]
        if len(matches) > 1:
            raise ValueError(
                "memory contains duplicate accepted trajectories for one training window"
            )
        return matches[0] if matches else None

    def _retain_memory_window(
        self,
        window,
        idx,
        output_path,
        source_fingerprint,
        build_fingerprint,
        verbose,
        source_origin_row=None,
    ):
        result = self.run_memory_window(window, window_idx=idx, verbose=verbose)
        if result.metrics.get("memory_save", 0.0) < 1.0:
            log_sample_progress(
                verbose,
                f"[sample {idx}] no accepted trajectory; memory_entries={len(self.memory.entries)}",
            )
            return {"retained": False, "window_idx": idx}
        aux = result.observations.get("model_auxiliary_tool", {})
        self.memory.add(
            target_history=window.target_history,
            tool_schedule=result.tool_schedule,
            diagnostic_outputs=result.observations,
            trajectory=result.reflection,
            metrics={"window_idx": float(idx), **result.metrics},
            input_json=series_to_json(
                window.history, window.target_col, window.timestamp_col
            ),
            ground_truth=(
                series_to_json(window.future, window.target_col, window.timestamp_col)
                if window.future is not None
                else ""
            ),
            forecast_full_prompt=result.forecast_full_prompt,
            forecast_system_prompt=result.forecast_system_prompt,
            forecast_answer=result.forecast_full_response,
            model_name=(
                ",".join(sorted(aux.get("best_model", {}).keys()))
                if isinstance(aux.get("best_model"), dict)
                else ""
            ),
            dataset_name=self.config.dataset_name,
        )
        entry = self.memory.entries[-1]
        entry["prompt_pack"] = prompt_identity(self.config)
        if source_origin_row is not None:
            entry["source_origin_row"] = source_origin_row
        entry["source_fingerprint"], entry["build_fingerprint"] = (
            source_fingerprint,
            build_fingerprint,
        )
        self.memory.save(output_path)
        log_sample_progress(
            verbose,
            f"[sample {idx}] accepted trajectory saved -> {output_path}; memory_entries={len(self.memory.entries)}",
        )
        return {"retained": True, "entry_hash": digest(entry), "window_idx": idx}

    def forecast_csv(
        self,
        data_path: str | Path,
        output_path: str | Path,
        stride: int | None = None,
        max_windows: int | None = None,
        latest: bool = False,
        show_progress: bool = False,
        covariate_forecasts_path: str | Path | None = None,
        covariate_issued_at: str | None = None,
        covariate_issue_col: str = "issued_at",
    ) -> ForecastResult | list[ForecastResult]:
        dataset = load_dataset(
            data_path, self.config.target_col, self.config.timestamp_col
        )
        forecasting = self.local_forecasting
        covariates = (
            pd.read_csv(covariate_forecasts_path, keep_default_na=False)
            if covariate_forecasts_path
            else None
        )
        windows = (
            []
            if latest
            else make_windows(
                dataset,
                lookback=self.config.lookback,
                horizon=self.config.horizon,
                stride=stride,
                max_windows=max_windows,
            )
        )
        if windows:
            results: list[ForecastResult] = []
            rows: list[dict[str, object]] = []
            for idx, window in enumerate(
                progress_iter(
                    windows,
                    label="Forecasting test windows",
                    unit="window",
                    enabled=show_progress,
                )
            ):
                if covariates is not None:
                    attach_covariate_forecasts(
                        window,
                        covariates,
                        self.config.horizon,
                        issued_at=covariate_issued_at,
                        issue_col=covariate_issue_col,
                    )
                result = self.run_window(
                    window,
                    forecasting=forecasting,
                    request_id=(
                        f"{self.config.execution_request_id}:window:{idx}"
                        if self.config.execution_request_id
                        else None
                    ),
                )
                results.append(result)
                aux = result.observations.get("model_auxiliary_tool", {})
                full_answer = result.forecast_full_response
                rows.append(
                    {
                        "idx": idx,
                        "model_name": (
                            ",".join(sorted(aux.get("best_model", {}).keys()))
                            if isinstance(aux.get("best_model"), dict)
                            else ""
                        ),
                        "answer": (
                            serialized_forecast(
                                result.forecast,
                                forecast_timestamps(window, self.config.horizon),
                                tags=False,
                            )
                            if result.forecast
                            else ""
                        ),
                        "status": result.status,
                        "failure_reason": result.failure_reason,
                        "attempts": result.attempts,
                        "fallback_source": result.fallback_source or "",
                        "ground_truth": (
                            series_to_json(
                                window.future, window.target_col, window.timestamp_col
                            )
                            if window.future is not None
                            else ""
                        ),
                        "prompt": result.forecast_full_prompt,
                        "system_prompt": result.forecast_system_prompt,
                        "prompt_mode": self.config.prompt_mode,
                        "prompt_fingerprint": self.config.prompt_fingerprint,
                        "full_answer": full_answer,
                        "mse": result.metrics.get("mse", ""),
                        "mae": result.metrics.get("mae", ""),
                        "tools": json.dumps(result.tool_schedule, ensure_ascii=False),
                        "reflection": result.reflection,
                    }
                )
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                atomic_write_csv(output_path, pd.DataFrame(rows))
            return results

        if not latest:
            raise ValueError(
                "no complete evaluation windows; pass latest=True explicitly for deployment forecasting"
            )
        window = make_latest_window(dataset, self.config.lookback)
        if covariates is not None:
            attach_covariate_forecasts(
                window,
                covariates,
                self.config.horizon,
                issued_at=covariate_issued_at,
                issue_col=covariate_issue_col,
            )
        result = self.run_window(
            window,
            forecasting=forecasting,
            request_id=(
                f"{self.config.execution_request_id}:latest"
                if self.config.execution_request_id
                else None
            ),
        )

        start_ts = None
        freq = None
        if dataset.timestamp_col:
            timestamps = pd.to_datetime(
                dataset.frame[dataset.timestamp_col], errors="coerce"
            )
            freq = (
                pd.infer_freq(timestamps.dropna())
                if timestamps.notna().sum() >= 3
                else None
            )
            if freq:
                start_ts = timestamps.iloc[-1] + pd.tseries.frequencies.to_offset(freq)
        write_forecast_csv(
            output_path, result.forecast, start_timestamp=start_ts, frequency=freq
        )
        return result


def progress_iter(
    items: Sequence[T], *, label: str, unit: str, enabled: bool
) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from items
    else:
        yield from tqdm(items, total=len(items), desc=label, unit=unit)


def completed_window_indices(
    memory: StrategyMemory, dataset_name: str | None = None
) -> set[int]:
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
    return (
        result_mse < config.excellent_mse_threshold
        and result_mae < config.excellent_mae_threshold
    )


def optional_tool_names(config: CastFlowConfig) -> list[str]:
    return optional_tools(config)


def deterministic_parallel_optional_strategies(
    state: WorkflowState,
    config: CastFlowConfig,
) -> list[list[str]]:
    strategies = [
        ["statistical_analysis_tool", "trend_analysis_tool", "changepoint_trend_tool"],
        [
            "basic_statistics_tool",
            "changepoint_trend_tool",
            "data_quality_tool",
            "event_summary_tool",
        ],
    ]
    if len(state.window.history.columns) > 2:
        strategies.append(
            ["cross_channel_tool", "event_summary_tool", "comprehensive_feature_tool"]
        )
    if config.allow_train_only_tools:
        strategies.append(
            [
                "autoregressive_residual_tool",
                "statistical_analysis_tool",
                "trend_analysis_tool",
            ]
        )
    return strategies


def metric_for_selection(result: ForecastResult) -> float:
    value = result.metrics.get("mse")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(score):
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


def serialized_forecast(
    values: list[float], timestamps: list[str], *, tags: bool = True
) -> str:
    keys = timestamps or [str(i) for i in range(len(values))]
    if len(keys) != len(values):
        raise ValueError("forecast timestamps and values have different lengths")
    answer = json.dumps(
        dict(zip(keys, map(float, values))), ensure_ascii=False, allow_nan=False
    )
    return f"<answer>{answer}</answer>" if tags else answer


def request_identity(
    window: TimeSeriesWindow, config: CastFlowConfig, model: ForecastingModule
) -> dict:
    from utils.prompting import prompt_identity

    root = Path(__file__).resolve().parents[1]
    sources = {}
    for folder in (
        "workflow",
        "forecasting",
        "utils",
        "anchorer",
        "backends",
        "evaluation",
    ):
        for path in sorted((root / folder).rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    def frame_identity(frame):
        if frame is None:
            return None
        return {
            "columns": list(map(str, frame.columns)),
            "dtypes": list(map(str, frame.dtypes)),
            "values_hash": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest(),
        }

    llm = getattr(model, "llm", None)
    anchor_identity = None
    if config.anchor_library_path:
        library_path = (
            Path(config.anchor_library_path).expanduser().resolve(strict=True)
        )
        if not library_path.is_file():
            raise ValueError(
                "anchor_library_path must reference an explicit library JSON file"
            )
        anchor_identity = {
            "path": str(library_path),
            "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        }
    return {
        "history": frame_identity(window.history),
        "prompt_pack": prompt_identity(config),
        "covariate_forecasts": frame_identity(window.covariate_forecasts),
        "covariate_issued_at": str(window.covariate_issued_at),
        "target": window.target_col,
        "timestamp": window.timestamp_col,
        "configuration_hash": digest(asdict(config)),
        "anchor_library": anchor_identity,
        "sources": sources,
        "forecasting_module": f"{type(model).__module__}.{type(model).__qualname__}",
        "forecasting_model": getattr(llm, "model", None),
        "forecasting_endpoint": getattr(llm, "base_url", None),
    }
