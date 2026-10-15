# OpenRouter provider

Instantiate `app.ai.open_router_provider.OpenRouterProvider(Settings())` in the
infrastructure/composition layer and pass it as `LLMProvider` to consumers. No
contract changes or protocol inheritance are required. Use `async with` to close
its HTTPX AsyncClient, or call `aclose()`. An injected AsyncClient is caller-owned.
When stopping a stream early, use `contextlib.aclosing()` on the iterator.

Copy configuration names from `.env.example` into the local environment. Global
settings allow missing credentials/models so the health endpoint and fake remain
usable. The provider requires a key at construction and validates the relevant
model before each operation. Timeout must be finite and positive, retries >= 0.

- `stream`: chat completions SSE translated to ordered strings, ignoring comments,
  role-only and usage events. `[DONE]` terminates; premature EOF is a transient
  failure. Multi-line SSE data is supported. The HTTP response closes on errors,
  cancellation and explicit generator close.
- `evaluate`: temperature 0 and strict JSON Schema for `is_correct`/`feedback`.
  The model must support structured outputs. This is only a minimal factual
  assessment of the supplied response; missing context yields false plus an
  explanation. Temperature 0 does not guarantee identical model outputs. No
  Socratic rubric, exercise context or TutorService is introduced here.
- `embed`: `/embeddings`, float encoding, returning a nonempty finite float vector.

Timeout applies to HTTPX connect/read/write/pool waits on every request, including
injected clients. Default: 30 seconds. It is an inactivity/phase timeout, not a
wall-clock deadline for a long stream or the entire retry sequence.

408, 429, recoverable 5xx (excluding 501/505), network errors, timeouts and remote
protocol failures map to `TransientProviderError`. Other unsuccessful statuses,
invalid configuration, malformed payloads and unsupported response formats map
to `PermanentProviderError`. Errors inside successful JSON/SSE are also mapped.
External bodies and messages are never copied into public error messages. HTTPX
transport causes are chained for diagnostics; do not log request headers.

Retries are extra attempts (default 2, total 3), only for transient errors.
`Retry-After` seconds and HTTP dates are honored when valid; otherwise delays are
0.25, 0.5, 1, 2 seconds, capped at 2 seconds. Sleep is injectable. No replay occurs
after delivering the first nonempty stream chunk, avoiding duplicate output.
Retries of a failed request can still cause upstream work/cost; there is no
exactly-once guarantee.

`last_metadata` is a bounded infrastructure diagnostic snapshot of the last
finished operation (including failures/explicit close). It records operation,
attempt count, elapsed wall time including retries and consumption, and upstream
provider/model, request header ID, generation ID and usage when supplied. Missing
values remain None/empty; no provider/model/token counts are fabricated. With
concurrent calls the last completed operation wins. It is not an audit store.

Tests use HTTPX MockTransport and fake byte streams. An autouse socket guard
forbids network connections/DNS; no real key, network or paid call is needed.
`httpx` is a runtime dependency; `httpx2` remains in dev for Starlette TestClient.

## Optional manual smoke test

After automated checks pass, set `SOKRA_OPENROUTER_API_KEY` and
`SOKRA_OPENROUTER_CHAT_MODEL` locally (or in ignored `services/api/.env`). Never
paste a key into a command, commit, issue or PR. Run from `services/api`:

```sh
uv run python - <<'PY'
import asyncio
from app.ai.open_router_provider import OpenRouterProvider
from app.ai.provider import ProviderError
from app.core.config import Settings

async def main():
    settings = Settings()
    if not settings.openrouter_api_key or not settings.openrouter_chat_model:
        print("Skipped: local API key/chat model missing")
        return
    settings.openrouter_retries = 0  # One deliberate billable request.
    try:
        async with OpenRouterProvider(settings) as provider:
            async for chunk in provider.stream("Reply with only OK."):
                print(chunk, end="", flush=True)
            print("\nCompleted")
    except ProviderError as exc:
        print(type(exc).__name__, str(exc))  # No HTTP headers or raw body.

asyncio.run(main())
PY
```

This manual command may incur cost and is never invoked by tests/CI. To check the
other operations manually, use `evaluate("2 + 2 = 4")` with a schema-capable model
or `embed("hello")` with `SOKRA_OPENROUTER_EMBEDDING_MODEL` configured.

Reference: [OpenRouter streaming](https://openrouter.ai/docs/api_reference/streaming)
and [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
