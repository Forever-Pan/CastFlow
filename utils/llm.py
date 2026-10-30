from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
import time
from email.utils import parsedate_to_datetime

from utils.config import CastFlowConfig


@dataclass(slots=True)
class LLMResponse:
    content: str
    ok: bool
    error: str = ""
    reasoning_content: str = ""
    finish_reason: str = ""


class OpenAICompatibleLLM:
    """Thin OpenAI-compatible chat client backed by CastFlow/.env."""

    def __init__(
        self,
        config: CastFlowConfig,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.config = config
        self.base_url = base_url if base_url is not None else config.api_base_url
        self.api_key = api_key if api_key is not None else config.api_key
        self.model = model if model is not None else config.api_model
        self.enabled = config.use_api if enabled is None else enabled
        self._client = None
        self._client_lock = threading.Lock()

    def available(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.config.validate()
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be finite and between 0 and 2")
        if not self.available():
            return LLMResponse("", ok=False, error="API is not configured")
        attempts = self.config.api_max_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not response.choices:
                    return LLMResponse(
                        "", ok=False, error="provider returned no choices"
                    )
                choice = response.choices[0]
                content = choice.message.content
                if not isinstance(content, str) or not content.strip():
                    return LLMResponse(
                        "", ok=False, error="provider returned no final answer"
                    )
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
                return LLMResponse(
                    content.strip(),
                    ok=True,
                    reasoning_content=reasoning if isinstance(reasoning, str) else "",
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                )
            except Exception as exc:
                last_error = self._safe_error(exc)
                status = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(exc, (ConnectionError, TimeoutError))
                    or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                    or status in {408, 409, 429}
                    or isinstance(status, int)
                    and 500 <= status < 600
                )
                if attempt >= attempts or not retryable:
                    break
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", {}) or {}
                delay = min(
                    8.0, self.config.api_backoff_seconds * 2 ** min(attempt - 1, 5)
                )
                hint = _retry_after(headers.get("retry-after"))
                if hint is not None and hint > 60:
                    last_error += (
                        "; server requested a longer delay; automatic retries stopped"
                    )
                    break
                time.sleep(max(delay, hint or 0.0))
        return LLMResponse(
            "", ok=False, error=f"{last_error} after {attempt} attempt(s)"
        )

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI-compatible API calls require the openai package"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.api_timeout,
                max_retries=0,
            )
        return self._client

    def _safe_error(self, exc: Exception) -> str:
        message = f"{type(exc).__name__}: {exc}"
        for secret in (
            self.api_key,
            self.config.api_key,
            self.config.local_model_api_key,
        ):
            if secret and secret != "EMPTY":
                message = message.replace(secret, "[redacted]")
        message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", message)
        return message[:1000]

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None
