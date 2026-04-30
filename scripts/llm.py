from __future__ import annotations

from dataclasses import dataclass

from .config import CastFlowConfig


@dataclass(slots=True)
class LLMResponse:
    content: str
    ok: bool
    error: str = ""


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

    def available(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self.available():
            return LLMResponse("", ok=False, error="API is not configured")
        attempts = max(1, int(self.config.api_max_retries) + 1)
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
                return LLMResponse((response.choices[0].message.content or "").strip(), ok=True)
            except Exception as exc:  # noqa: BLE001 - propagate provider error text into fallback path
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= attempts:
                    break
        return LLMResponse("", ok=False, error=f"{last_error} after {attempts} attempt(s)")

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("OpenAI-compatible API calls require the openai package") from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.api_timeout,
            )
        return self._client
