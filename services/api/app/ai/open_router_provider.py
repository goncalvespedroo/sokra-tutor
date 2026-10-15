"""OpenRouter wire formats and transport stay inside this infrastructure adapter."""

import asyncio
import json
import math
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from time import perf_counter, time
from typing import Any

import httpx

from app.ai.provider import (
    EvaluationResult,
    PermanentProviderError,
    TransientProviderError,
)
from app.core.config import Settings


@dataclass
class RequestMetadata:
    """Last finished operation; absent upstream values remain None."""

    operation: str
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    generation_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    attempts: int = 0

    def update(self, data: dict[str, Any]) -> None:
        for key in ("provider", "model"):
            if isinstance(data.get(key), str):
                setattr(self, key, data[key])
        if isinstance(data.get("id"), str):
            self.generation_id = data["id"]
        if isinstance(data.get("usage"), dict):
            self.usage.update(data["usage"])


def _error(code: object) -> PermanentProviderError | TransientProviderError:
    transient = code in (408, 429, "server_error", "rate_limit_exceeded") or (
        isinstance(code, int) and 500 <= code <= 599 and code not in (501, 505)
    )
    cls = TransientProviderError if transient else PermanentProviderError
    # Never echo upstream bodies, prompts, credentials or arbitrary error messages.
    return cls(
        "Temporary provider failure" if transient else "Provider rejected request"
    )


def _retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = parsedate_to_datetime(value).timestamp() - time()
            except ValueError, TypeError, OverflowError:
                delay = float("nan")
        if math.isfinite(delay):
            return max(0.0, delay)
    return min(0.25 * 2 ** min(attempt, 4), 2.0)


def _object(raw: str | bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError) as exc:
        raise PermanentProviderError("Invalid provider response") from exc


def _check_data(data: dict[str, Any]) -> None:
    if "error" in data:
        error = data["error"]
        raise _error(error.get("code") if isinstance(error, dict) else None)


class OpenRouterProvider:
    """Structurally implements LLMProvider. Close owned clients with async with.

    Injected clients remain the caller's responsibility. last_metadata is a bounded
    diagnostic snapshot: concurrent calls replace it in completion order.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        key = settings.openrouter_api_key
        try:
            url = httpx.URL(settings.openrouter_base_url)
            if (
                url.scheme not in ("http", "https")
                or not url.host
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError
        except (ValueError, httpx.InvalidURL) as exc:
            raise PermanentProviderError("Invalid provider base URL") from exc
        if (
            not key
            or not key.strip()
            or any(c.isspace() for c in key)
            or not key.isascii()
        ):
            raise PermanentProviderError(
                "Provider API key is required and must be valid"
            )
        self._base_url = str(url).rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._chat_model = settings.openrouter_chat_model
        self._embedding_model = settings.openrouter_embedding_model
        self._timeout = settings.openrouter_timeout_seconds
        self._retries = settings.openrouter_retries
        self._owned = client is None
        self._client = client if client is not None else httpx.AsyncClient()
        self._sleep = sleep
        self.last_metadata: RequestMetadata | None = None

    async def __aenter__(self) -> "OpenRouterProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()

    @staticmethod
    def _model(value: str | None) -> str:
        if not value or not value.strip():
            raise PermanentProviderError("Provider model is required")
        return value

    async def _events(
        self, response: httpx.Response
    ) -> AsyncGenerator[dict[str, Any], None]:
        lines: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                lines.append(line[5:].removeprefix(" "))
            elif not line and lines:
                raw = "\n".join(lines)
                lines.clear()
                if raw == "[DONE]":
                    return
                yield _object(raw)
        # Missing terminator means a truncated response, even after finish_reason.
        raise TransientProviderError("Provider stream ended unexpectedly")

    async def _request(
        self,
        operation: str,
        endpoint: str,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        started = perf_counter()
        metadata = RequestMetadata(operation)
        delivered = False
        try:
            for attempt in range(self._retries + 1):
                metadata.attempts = attempt + 1
                retry_after = None
                try:
                    async with self._client.stream(
                        "POST",
                        f"{self._base_url}/{endpoint}",
                        headers=self._headers,
                        json=payload,
                        timeout=self._timeout,
                        follow_redirects=False,
                    ) as response:
                        metadata.request_id = response.headers.get("x-request-id")
                        metadata.generation_id = response.headers.get("x-generation-id")
                        retry_after = response.headers.get("retry-after")
                        if not 200 <= response.status_code < 300:
                            raise _error(response.status_code)
                        if stream:
                            async for data in self._events(response):
                                metadata.update(data)
                                _check_data(data)
                                # Only useful content prevents replay on failure.
                                content = self._content(data)
                                if content:
                                    delivered = True
                                yield data
                        else:
                            data = _object(await response.aread())
                            metadata.update(data)
                            _check_data(data)
                            yield data
                    return
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                ) as exc:
                    error = TransientProviderError("Provider transport unavailable")
                    if delivered or attempt == self._retries:
                        raise error from exc
                except TransientProviderError:
                    if delivered or attempt == self._retries:
                        raise
                except httpx.HTTPError as exc:
                    raise PermanentProviderError("Provider transport failure") from exc
                # Small bounded deterministic backoff; sleep is injectable in tests.
                await self._sleep(_retry_delay(retry_after, attempt))
                metadata = RequestMetadata(operation)
        finally:
            metadata.latency_seconds = perf_counter() - started
            self.last_metadata = metadata

    @staticmethod
    def _content(data: dict[str, Any]) -> str | None:
        try:
            choices = data.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError
            if not choices:
                return None
            choice = choices[0]
            if choice.get("finish_reason") == "error":
                raise TransientProviderError("Provider generation failed")
            content = choice["delta"].get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError
            return content
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PermanentProviderError("Invalid provider stream content") from exc

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        payload = {
            "model": self._model(self._chat_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        events = self._request("stream", "chat/completions", payload, stream=True)
        try:
            async for data in events:
                content = self._content(data)
                if content:
                    yield content
        finally:
            await events.aclose()

    async def evaluate(self, response: str) -> EvaluationResult:
        payload = {
            "model": self._model(self._chat_model),
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Evaluate the factual correctness of the supplied response "
                        "on its own. Return is_correct (boolean) "
                        "and feedback (string). "
                        "If context is insufficient, return false and explain this."
                    ),
                },
                {"role": "user", "content": response},
            ],
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_correct": {"type": "boolean"},
                            "feedback": {"type": "string"},
                        },
                        "required": ["is_correct", "feedback"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        result = None
        async with aclosing(
            self._request("evaluate", "chat/completions", payload)
        ) as responses:
            async for data in responses:
                try:
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError
                    parsed = _object(content)
                    if (
                        type(parsed.get("is_correct")) is not bool
                        or not isinstance(parsed.get("feedback"), str)
                        or set(parsed) != {"is_correct", "feedback"}
                    ):
                        raise ValueError
                    result = EvaluationResult(parsed["is_correct"], parsed["feedback"])
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise PermanentProviderError("Invalid provider evaluation") from exc
        assert result is not None
        return result

    async def embed(self, text: str) -> list[float]:
        payload = {
            "model": self._model(self._embedding_model),
            "input": text,
            "encoding_format": "float",
        }
        result: list[float] = []
        async with aclosing(self._request("embed", "embeddings", payload)) as responses:
            async for data in responses:
                try:
                    vector = data["data"][0]["embedding"]
                    if not isinstance(vector, list) or not vector:
                        raise ValueError
                    if any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in vector
                    ):
                        raise ValueError
                    result = [float(v) for v in vector]
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                    OverflowError,
                ) as exc:
                    raise PermanentProviderError("Invalid provider embedding") from exc
        return result
