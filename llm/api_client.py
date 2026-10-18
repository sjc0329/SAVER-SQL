import json
import http.client
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from evaluation.cost_tracker import Usage, usage_from_api


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    raw: dict[str, Any]


class LLMClient:
    def __init__(self, config: dict[str, Any], prices: dict[str, float] | None = None):
        env_base = config.get("api_base_url_env", "SAVER_API_BASE_URL")
        env_key = config.get("api_key_env", "SAVER_API_KEY")
        env_model = config.get("model_env", "SAVER_MODEL")
        self.base_url = os.environ.get(env_base, "").rstrip("/")
        self.api_key = os.environ.get(env_key, "")
        self.model = os.environ.get(env_model, config.get("default_model", "deepseek-chat"))
        self.max_tokens = int(config.get("max_tokens", 1200))
        self.thinking = config.get("thinking")
        self.reasoning_effort = config.get("reasoning_effort")
        self.use_response_format = bool(config.get("use_response_format", True))
        self.timeout_seconds = int(config.get("request_timeout_seconds", 120))
        self.request_retries = int(config.get("request_retries", 3))
        self.retry_backoff_seconds = float(config.get("retry_backoff_seconds", 2.0))
        self.prices = prices or {}
        self.ssl_context = self._build_ssl_context()

    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.configured():
            raise RuntimeError("LLM is not configured. Set SAVER_API_BASE_URL, SAVER_API_KEY, and SAVER_MODEL.")

        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if response_format is not None and self.use_response_format:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        data = json.dumps(payload).encode("utf-8")
        raw = self._post_json_with_retries(url, data)

        text = raw["choices"][0]["message"]["content"]
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        usage = usage_from_api(raw.get("usage"), prompt_text, text, self.prices)
        return LLMResponse(text=text, usage=usage, raw=raw)

    def _post_json_with_retries(self, url: str, data: bytes) -> dict[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(self.request_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds, context=self.ssl_context) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if not self._retryable_http_status(exc.code) or attempt >= self.request_retries:
                    raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc
                last_error = RuntimeError(f"LLM HTTP error {exc.code}: {body[:300]}")
            except (
                TimeoutError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                ssl.SSLError,
            ) as exc:
                if attempt >= self.request_retries:
                    raise RuntimeError(f"LLM request failed after {attempt + 1} attempts: {exc}") from exc
                last_error = exc
            self._sleep_before_retry(attempt, last_error)
        raise RuntimeError(f"LLM request failed: {last_error}")

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        cafile = os.environ.get("SAVER_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
        if cafile:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.load_verify_locations(cafile=cafile)
            return context
        try:
            import certifi  # type: ignore

            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.load_verify_locations(cafile=certifi.where())
            return context
        except Exception:
            return ssl.create_default_context()

    @staticmethod
    def _retryable_http_status(status: int) -> bool:
        return status in {408, 409, 425, 429} or status >= 500

    def _sleep_before_retry(self, attempt: int, error: BaseException | None) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        print(f"LLM request failed ({error}); retrying in {delay:.1f}s...", file=sys.stderr, flush=True)
        time.sleep(delay)
