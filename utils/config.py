from __future__ import annotations

import os
from dataclasses import dataclass
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
    api_key: str | None = None
    api_model: str | None = None
    local_model_base_url: str | None = None
    local_model_name: str | None = None
    local_model_api_key: str | None = "EMPTY"
    api_timeout: float = 60.0
    api_max_retries: int = 2
    planner_max_tokens: int = 1200
    forecasting_max_tokens: int = 7000
    reflection_max_tokens: int = 800
    excellent_mse_threshold: float = 168.0
    excellent_mae_threshold: float = 8.6

    def validate(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")
        if self.memory_top_k <= 0:
            raise ValueError("memory_top_k must be positive")
        if self.parallel_plan_k <= 0:
            raise ValueError("parallel_plan_k must be positive")

    @classmethod
    def from_env(cls, **overrides: object) -> "CastFlowConfig":
        values = load_castflow_env()
        cfg = cls(
            use_api=parse_bool(values.get("CASTFLOW_USE_API", values.get("USE_API", "true"))),
            api_base_url=values.get("OPENAI_BASE_URL"),
            api_key=values.get("OPENAI_API_KEY"),
            api_model=values.get("MODEL"),
            local_model_base_url=values.get("LOCAL_MODEL_BASE_URL"),
            local_model_name=values.get("LOCAL_MODEL_NAME"),
            local_model_api_key=values.get("LOCAL_MODEL_API_KEY", "EMPTY"),
            dataset_name=values.get("CASTFLOW_DATASET_NAME"),
            anchor_library_path=values.get("CASTFLOW_ANCHOR_LIBRARY"),
            max_retry_loops=int(values.get("MAX_REFLECTION_LOOPS", values.get("CASTFLOW_MAX_RETRY_LOOPS", 3))),
            test_max_retry_loops=int(values.get("MAX_TEST_RETRY_LOOPS", values.get("CASTFLOW_TEST_MAX_RETRY_LOOPS", 10))),
            parallel_plan_k=int(values.get("PARALLEL_PLAN_K", values.get("CASTFLOW_PARALLEL_PLAN_K", 4))),
            api_timeout=float(values.get("CASTFLOW_API_TIMEOUT", 60.0)),
            api_max_retries=int(values.get("CASTFLOW_API_MAX_RETRIES", 2)),
            planner_max_tokens=int(values.get("CASTFLOW_PLANNER_MAX_TOKENS", 1200)),
            forecasting_max_tokens=int(values.get("CASTFLOW_FORECASTING_MAX_TOKENS", 7000)),
            reflection_max_tokens=int(values.get("CASTFLOW_REFLECTION_MAX_TOKENS", 800)),
            excellent_mse_threshold=float(values.get("EXCELLENT_MSE_THRESHOLD", values.get("CASTFLOW_EXCELLENT_MSE_THRESHOLD", 168.0))),
            excellent_mae_threshold=float(values.get("EXCELLENT_MAE_THRESHOLD", values.get("CASTFLOW_EXCELLENT_MAE_THRESHOLD", 8.6))),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    def api_ready(self) -> bool:
        return bool(self.use_api and self.api_base_url and self.api_key and self.api_model)

    def local_forecast_ready(self) -> bool:
        return bool(self.use_api and self.local_model_base_url and self.local_model_name)


def load_castflow_env() -> dict[str, str]:
    """Load CastFlow/.env plus process environment without exposing secrets."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = unquote_env_value(value.strip())
    values.update({key: value for key, value in os.environ.items() if key.startswith(("OPENAI_", "MODEL", "LOCAL_MODEL_", "CASTFLOW_", "USE_API"))})
    return values


def unquote_env_value(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
