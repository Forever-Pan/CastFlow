from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(slots=True)
class CastFlowConfig:
    """Runtime configuration aligned with the CastFlow paper terminology."""

    lookback: int = 96
    horizon: int = 96
    seasonal_period: int = 24
    memory_top_k: int = 3
    memory_similarity_threshold: float = 0.90
    max_retry_loops: int = 3
    test_max_retry_loops: int = 10
    parallel_plan_k: int = 4
    target_col: str | None = None
    timestamp_col: str | None = None
    dataset_name: str | None = None
    anchor_library_path: str | None = None
    allow_train_only_tools: bool = False
    use_api: bool = True
    api_base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_model: str | None = None
    local_model_base_url: str | None = None
    local_model_name: str | None = None
    local_model_api_key: str | None = field(default="EMPTY", repr=False)
    api_timeout: float = 60.0
    api_max_retries: int = 2
    api_backoff_seconds: float = 0.5
    planner_max_tokens: int = 1200
    forecasting_max_tokens: int = 5000
    reflection_max_tokens: int = 800
    forecasting_temperature: float = 0.3
    diagnostic_cache_policy: str = "current_plan"
    execution_journal_dir: str | None = None
    execution_request_id: str | None = None
    forecasting_model_revision: str | None = None
    reasoning_model_revision: str | None = None
    prompt_dir: str | None = None
    prompt_mode: str | None = None
    prompt_fingerprint: str | None = None
    excellent_mse_threshold: float = 168.0
    excellent_mae_threshold: float = 8.6

    def validate(self) -> None:
        positive = (
            "lookback",
            "horizon",
            "seasonal_period",
            "memory_top_k",
            "parallel_plan_k",
            "planner_max_tokens",
            "forecasting_max_tokens",
            "reflection_max_tokens",
        )
        for name in positive + (
            "max_retry_loops",
            "test_max_retry_loops",
            "api_max_retries",
        ):
            value = getattr(self, name)
            minimum = 1 if name in positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.lookback < 2:
            raise ValueError("lookback must be greater than 1")
        for name in (
            "api_timeout",
            "api_backoff_seconds",
            "forecasting_temperature",
            "memory_similarity_threshold",
            "excellent_mse_threshold",
            "excellent_mae_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.api_timeout <= 0 or self.api_backoff_seconds < 0:
            raise ValueError(
                "API timeout must be positive and backoff must be nonnegative"
            )
        if not 0 <= self.forecasting_temperature <= 2:
            raise ValueError("forecasting_temperature must be between 0 and 2")
        if not -1 <= self.memory_similarity_threshold <= 1:
            raise ValueError("memory_similarity_threshold must be between -1 and 1")
        if self.excellent_mse_threshold < 0 or self.excellent_mae_threshold < 0:
            raise ValueError("error thresholds must be nonnegative")
        if self.diagnostic_cache_policy not in {
            "none",
            "all_primitives",
            "current_plan",
        }:
            raise ValueError(
                "diagnostic_cache_policy must be none, all_primitives, or current_plan"
            )
        if not isinstance(self.use_api, bool) or not isinstance(
            self.allow_train_only_tools, bool
        ):
            raise ValueError("API and training-only tool switches must be booleans")
        from utils.prompting import validate_prompt_config

        validate_prompt_config(self)

    @classmethod
    def from_env(cls, **overrides: object) -> "CastFlowConfig":
        values = load_castflow_env()
        cfg = cls(
            use_api=parse_bool(
                values.get("CASTFLOW_USE_API", values.get("USE_API", "true"))
            ),
            api_base_url=values.get("OPENAI_BASE_URL") or None,
            api_key=values.get("OPENAI_API_KEY") or None,
            api_model=values.get("MODEL") or None,
            local_model_base_url=values.get("LOCAL_MODEL_BASE_URL") or None,
            local_model_name=values.get("LOCAL_MODEL_NAME") or None,
            local_model_api_key=values.get("LOCAL_MODEL_API_KEY") or "EMPTY",
            dataset_name=values.get("CASTFLOW_DATASET_NAME"),
            anchor_library_path=values.get("CASTFLOW_ANCHOR_LIBRARY"),
            max_retry_loops=int(
                values.get(
                    "MAX_REFLECTION_LOOPS", values.get("CASTFLOW_MAX_RETRY_LOOPS", 3)
                )
            ),
            test_max_retry_loops=int(
                values.get(
                    "MAX_TEST_RETRY_LOOPS",
                    values.get("CASTFLOW_TEST_MAX_RETRY_LOOPS", 10),
                )
            ),
            parallel_plan_k=int(
                values.get("PARALLEL_PLAN_K", values.get("CASTFLOW_PARALLEL_PLAN_K", 4))
            ),
            api_timeout=float(values.get("CASTFLOW_API_TIMEOUT", 60.0)),
            api_max_retries=int(values.get("CASTFLOW_API_MAX_RETRIES", 2)),
            api_backoff_seconds=float(values.get("CASTFLOW_API_BACKOFF_SECONDS", 0.5)),
            memory_top_k=int(values.get("CASTFLOW_MEMORY_TOP_K", 3)),
            memory_similarity_threshold=float(
                values.get("CASTFLOW_MEMORY_THRESHOLD", 0.90)
            ),
            planner_max_tokens=int(values.get("CASTFLOW_PLANNER_MAX_TOKENS", 1200)),
            forecasting_max_tokens=int(
                values.get("CASTFLOW_FORECASTING_MAX_TOKENS", 5000)
            ),
            reflection_max_tokens=int(
                values.get("CASTFLOW_REFLECTION_MAX_TOKENS", 800)
            ),
            forecasting_temperature=float(
                values.get("CASTFLOW_FORECASTING_TEMPERATURE", 0.3)
            ),
            diagnostic_cache_policy=values.get(
                "CASTFLOW_DIAGNOSTIC_CACHE_POLICY", "current_plan"
            ),
            execution_journal_dir=values.get("CASTFLOW_EXECUTION_JOURNAL_DIR") or None,
            execution_request_id=values.get("CASTFLOW_REQUEST_ID") or None,
            forecasting_model_revision=values.get("CASTFLOW_FORECASTING_MODEL_REVISION")
            or None,
            reasoning_model_revision=values.get("CASTFLOW_REASONING_MODEL_REVISION")
            or None,
            prompt_dir=values.get("CASTFLOW_PROMPT_DIR") or None,
            excellent_mse_threshold=float(
                values.get(
                    "EXCELLENT_MSE_THRESHOLD",
                    values.get("CASTFLOW_EXCELLENT_MSE_THRESHOLD", 168.0),
                )
            ),
            excellent_mae_threshold=float(
                values.get(
                    "EXCELLENT_MAE_THRESHOLD",
                    values.get("CASTFLOW_EXCELLENT_MAE_THRESHOLD", 8.6),
                )
            ),
        )
        unknown = set(overrides) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"unknown configuration fields: {', '.join(sorted(unknown))}"
            )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        cfg.validate()
        return cfg

    def api_ready(self) -> bool:
        return bool(
            self.use_api and self.api_base_url and self.api_key and self.api_model
        )

    def local_forecast_ready(self) -> bool:
        return bool(
            self.use_api and self.local_model_base_url and self.local_model_name
        )


def load_castflow_env() -> dict[str, str]:
    """Load CastFlow/.env plus process environment without exposing secrets."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for number, line in enumerate(
            env_path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            if "=" not in stripped:
                raise ValueError(f"invalid .env assignment on line {number}")
            key, value = stripped.split("=", 1)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                raise ValueError(f"invalid .env key on line {number}")
            values[key.strip()] = unquote_env_value(value.strip())
    legacy_keys = {
        "MAX_REFLECTION_LOOPS",
        "MAX_TEST_RETRY_LOOPS",
        "PARALLEL_PLAN_K",
        "EXCELLENT_MSE_THRESHOLD",
        "EXCELLENT_MAE_THRESHOLD",
        "MODEL",
        "USE_API",
    }
    values.update(
        {
            key: value
            for key, value in os.environ.items()
            if key in legacy_keys
            or key.startswith(("OPENAI_", "LOCAL_MODEL_", "CASTFLOW_"))
        }
    )
    return values


def unquote_env_value(value: str) -> str:
    if value.startswith(('"', "'")):
        end = value.find(value[0], 1)
        if end < 0 or (
            value[end + 1 :].strip() and not value[end + 1 :].lstrip().startswith("#")
        ):
            raise ValueError("invalid quoted .env value")
        return value[1:end]
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean settings must be true/false, yes/no, on/off, or 1/0")
