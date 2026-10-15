import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing

import httpx
import pytest
from pydantic import ValidationError

from app.ai import (
    EvaluationResult,
    LLMProvider,
    PermanentProviderError,
    TransientProviderError,
)
from app.ai.open_router_provider import OpenRouterProvider
from app.core.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Settings:
    monkeypatch.chdir(tmp_path)
    for field in Settings.model_fields:
        monkeypatch.delenv(f"SOKRA_{field.upper()}", raising=False)
    return Settings(
        openrouter_api_key="test-only-key",
        openrouter_chat_model="test/chat",
        openrouter_embedding_model="test/embed",
        openrouter_timeout_seconds=7.5,
    )


async def no_sleep(_: float) -> None:
    pass


def event(data: object) -> str:
    return f"data: {json.dumps(data)}\n\n"


def success(operation: str) -> httpx.Response:
    if operation == "stream":
        return httpx.Response(
            200,
            text=event({"choices": [{"delta": {"content": "ok"}}]})
            + "data: [DONE]\n\n",
        )
    if operation == "evaluate":
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"is_correct":true,"feedback":"ok"}'}}
                ]
            },
        )
    return httpx.Response(200, json={"data": [{"embedding": [1, 0.5]}]})


async def invoke(provider: LLMProvider, operation: str):
    if operation == "stream":
        return [part async for part in provider.stream("input")]
    if operation == "evaluate":
        return await provider.evaluate("input")
    return await provider.embed("input")


def run(settings: Settings, handler: Callable, operation: str = "embed"):
    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider: LLMProvider = OpenRouterProvider(
                settings, client=client, sleep=no_sleep
            )
            return await invoke(provider, operation)

    return asyncio.run(main())


def test_construction_contract_and_lifecycle(settings: Settings) -> None:
    async def main():
        async with OpenRouterProvider(settings) as provider:
            contract: LLMProvider = provider
            assert contract is provider
            assert provider.last_metadata is None
        assert provider._client.is_closed
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: success("embed"))
        ) as client:
            async with OpenRouterProvider(settings, client=client):
                pass
            assert not client.is_closed

    asyncio.run(main())
    assert "test-only-key" not in repr(settings)


@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
def test_requests(settings: Settings, operation: str) -> None:
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer test-only-key"
        assert request.headers["content-type"] == "application/json"
        assert request.extensions["timeout"] == dict.fromkeys(
            ["connect", "read", "write", "pool"], 7.5
        )
        body = json.loads(request.content)
        assert body["model"] == ("test/embed" if operation == "embed" else "test/chat")
        assert request.url.path == (
            "/api/v1/embeddings" if operation == "embed" else "/api/v1/chat/completions"
        )
        if operation == "evaluate":
            assert body["temperature"] == 0
            assert body["response_format"]["json_schema"]["strict"] is True
        elif operation == "stream":
            assert body["stream"] is True
            assert body["messages"] == [{"role": "user", "content": "input"}]
        else:
            assert body["input"] == "input"
        return success(operation)

    result = run(settings, handler, operation)
    if operation == "embed":
        assert result == [1.0, 0.5]
        assert isinstance(result, list)
        assert all(type(v) is float for v in result)
    elif operation == "evaluate":
        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True
        assert result.feedback == "ok"
    else:
        assert result == ["ok"]


def test_stream_and_metadata(settings: Settings) -> None:
    wire = ": heartbeat\r\n\r\n" + event(
        {"choices": [{"delta": {"role": "assistant"}}]}
    )
    wire += (
        'event: message\ndata: {"choices":\ndata: [{"delta":{"content":"Olá"}}]}\n\n'
    )
    wire += event({"choices": [{"delta": {"content": " mundo"}}]})
    wire += event(
        {
            "choices": [],
            "id": "gen-1",
            "provider": "upstream",
            "model": "actual/model",
            "usage": {"total_tokens": 12},
        }
    )
    wire += "data: [DONE]\n\n"

    async def main():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, text=wire, headers={"x-request-id": "req-1"}
                )
            )
        ) as client:
            provider = OpenRouterProvider(settings, client=client)
            assert await invoke(provider, "stream") == ["Olá", " mundo"]
            meta = provider.last_metadata
            assert meta is not None
            assert (meta.provider, meta.model, meta.request_id, meta.generation_id) == (
                "upstream",
                "actual/model",
                "req-1",
                "gen-1",
            )
            assert meta.usage == {"total_tokens": 12}
            assert meta.latency_seconds >= 0
            assert meta.attempts == 1

    asyncio.run(main())


@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
@pytest.mark.parametrize(
    "failure", [429, 408, 500, 502, 503, 504, "timeout", "connect", "protocol"]
)
def test_transient_retry(
    settings: Settings, operation: str, failure: int | str
) -> None:
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls > 1:
            return success(operation)
        errors = {
            "timeout": httpx.ReadTimeout,
            "connect": httpx.ConnectError,
            "protocol": httpx.RemoteProtocolError,
        }
        if isinstance(failure, str):
            raise errors[failure]("transport failure", request=request)
        assert isinstance(failure, int)
        return httpx.Response(failure)

    run(settings, handler, operation)
    assert calls == 2


@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 501, 505, 302])
def test_permanent_no_retry(settings: Settings, operation: str, status: int) -> None:
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(
            status, text="test-only-key", headers={"location": "https://other.invalid"}
        )

    with pytest.raises(PermanentProviderError) as error:
        run(settings, handler, operation)
    assert "test-only-key" not in str(error.value)
    assert calls == 1


@pytest.mark.parametrize("retries", [0, 1, 2])
@pytest.mark.parametrize("failure", [429, 503, "timeout", "connect"])
def test_retry_budget(settings: Settings, retries: int, failure: int | str) -> None:
    settings.openrouter_retries = retries
    calls = 0
    delays = []

    async def sleep(delay):
        delays.append(delay)

    def handler(request):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("network", request=request)
        if failure == "connect":
            raise httpx.ConnectError("network", request=request)
        assert isinstance(failure, int)
        return httpx.Response(failure)

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterProvider(settings, client=client, sleep=sleep)
            with pytest.raises(TransientProviderError) as error:
                await provider.embed("text")
            if isinstance(failure, str):
                assert isinstance(error.value.__cause__, httpx.TransportError)
            assert provider.last_metadata is not None
            assert provider.last_metadata.attempts == retries + 1

    asyncio.run(main())
    assert calls == retries + 1
    assert delays == [0.25, 0.5][:retries]


@pytest.mark.parametrize(
    "operation,body",
    [
        ("embed", "not json"),
        ("embed", "[]"),
        ("embed", "{}"),
        ("embed", '{"data":[]}'),
        ("embed", '{"data":[{"embedding":[true]}]}'),
        ("embed", '{"data":[{"embedding":[NaN]}]}'),
        ("embed", '{"data":[{"embedding":[]}]}'),
        ("evaluate", "{}"),
        ("evaluate", '{"choices":[]}'),
        ("evaluate", '{"choices":[{"message":{"content":null}}]}'),
        (
            "evaluate",
            json.dumps(
                {
                    "choices": [
                        {"message": {"content": '{"is_correct":"true","feedback":"x"}'}}
                    ]
                }
            ),
        ),
        ("stream", "data: invalid\n\n"),
        ("stream", event({"choices": [{"delta": {"content": 1}}]})),
        ("stream", event({"choices": [None]})),
    ],
)
def test_malformed(settings: Settings, operation: str, body: str) -> None:
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=body)

    with pytest.raises(PermanentProviderError):
        run(settings, handler, operation)
    assert calls == 1


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("ending", ["eof", "error", "network"])
def test_stream_failure_replay_boundary(
    settings: Settings, partial: bool, ending: str
) -> None:
    calls = 0

    class Wire(httpx.AsyncByteStream):
        async def __aiter__(self):
            if partial:
                yield event({"choices": [{"delta": {"content": "first"}}]}).encode()
            if ending == "error":
                yield event({"error": {"code": "server_error"}}).encode()
            elif ending == "network":
                raise httpx.ReadError("network")

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=Wire())

    with pytest.raises(TransientProviderError):
        run(settings, handler, "stream")
    assert calls == (1 if partial else 3)


@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
@pytest.mark.parametrize(
    "code,error", [(401, PermanentProviderError), (503, TransientProviderError)]
)
def test_errors_inside_success_response(
    settings: Settings, operation: str, code: int, error: type[Exception]
) -> None:
    data = {"error": {"code": code, "message": "test-only-key"}}

    def handler(_):
        return (
            httpx.Response(200, text=event(data))
            if operation == "stream"
            else httpx.Response(200, json=data)
        )

    with pytest.raises(error, match="provider|Provider"):
        run(settings, handler, operation)


@pytest.mark.parametrize(
    "field,value",
    [
        ("openrouter_api_key", None),
        ("openrouter_api_key", "bad\nkey"),
        ("openrouter_base_url", "bad"),
        ("openrouter_base_url", "https://user:pass@example.com"),
    ],
)
def test_bad_configuration(settings: Settings, field: str, value) -> None:
    setattr(settings, field, value)
    with pytest.raises(PermanentProviderError):
        OpenRouterProvider(settings)


@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
def test_missing_model(settings: Settings, operation: str) -> None:
    settings.openrouter_chat_model = None
    settings.openrouter_embedding_model = None

    def handler(_):
        pytest.fail("No request expected")

    with pytest.raises(PermanentProviderError):
        run(settings, handler, operation)


@pytest.mark.parametrize(
    "field,value",
    [
        ("openrouter_timeout_seconds", 0),
        ("openrouter_timeout_seconds", -1),
        ("openrouter_timeout_seconds", float("inf")),
        ("openrouter_retries", -1),
    ],
)
def test_invalid_limits(settings: Settings, field: str, value) -> None:
    with pytest.raises(ValidationError):
        Settings(**(settings.model_dump() | {field: value}))


@pytest.mark.parametrize(
    "header,expected",
    [
        ("3", 3.0),
        ("invalid", 0.25),
        ("NaN", 0.25),
        ("Thu, 01 Jan 1970 00:00:00 GMT", 0.0),
    ],
)
def test_retry_after(settings: Settings, header: str, expected: float) -> None:
    delays = []

    async def sleep(delay):
        delays.append(delay)

    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, headers={"retry-after": header})
            if calls == 1
            else success("embed")
        )

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await OpenRouterProvider(settings, client=client, sleep=sleep).embed("text")

    asyncio.run(main())
    assert delays == [expected]


def test_stream_close_releases_response(settings: Settings) -> None:
    closed = False

    class Wire(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({"choices": [{"delta": {"content": "first"}}]}).encode()
            raise AssertionError("Consumer stopped; do not read again")

        async def aclose(self):
            nonlocal closed
            closed = True

    async def main():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Wire()))
        ) as client:
            provider = OpenRouterProvider(settings, client=client)
            async with aclosing(provider.stream("text")) as stream:
                assert await anext(stream) == "first"
            assert closed
            assert provider.last_metadata is not None

    asyncio.run(main())


@pytest.mark.parametrize("operation", ["evaluate", "embed"])
def test_nonstream_metadata(settings: Settings, operation: str) -> None:
    data = success(operation).json() | {
        "model": "actual",
        "provider": "upstream",
        "id": "generation",
        "usage": {"prompt_tokens": 5},
    }

    async def main():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
        ) as client:
            provider = OpenRouterProvider(settings, client=client)
            await invoke(provider, operation)
            meta = provider.last_metadata
            assert meta is not None
            assert meta.model == "actual"
            assert meta.provider == "upstream"
            assert meta.generation_id == "generation"
            assert meta.request_id is None
            assert meta.usage == {"prompt_tokens": 5}

    asyncio.run(main())
