import asyncio

import pytest

from app.ai import (
    EvaluationResult,
    FakeLLMProvider,
    LLMProvider,
    PermanentProviderError,
    ProviderError,
    TransientProviderError,
)


def test_fake_satisfies_provider_contract() -> None:
    fake = FakeLLMProvider()
    provider: LLMProvider = fake

    assert provider is fake


def test_stream_returns_configured_chunks_in_order() -> None:
    chunks = ["Olá", ", ", "aluno"]
    provider: LLMProvider = FakeLLMProvider(chunks=chunks)

    async def collect_chunks() -> list[str]:
        return [chunk async for chunk in provider.stream("Uma pergunta")]

    assert asyncio.run(collect_chunks()) == chunks


@pytest.mark.parametrize("is_correct", [True, False])
def test_evaluate_returns_configured_result(is_correct: bool) -> None:
    result = EvaluationResult(is_correct=is_correct, feedback="Feedback configurado")
    provider: LLMProvider = FakeLLMProvider(evaluation=result)

    async def evaluate() -> EvaluationResult:
        return await provider.evaluate("Uma resposta")

    actual = asyncio.run(evaluate())

    assert actual is result
    assert actual.is_correct is is_correct
    assert actual.feedback == result.feedback


def test_embed_returns_configured_vector() -> None:
    vector = [0.25, -0.5, 0.0]
    provider: LLMProvider = FakeLLMProvider(embedding=vector)

    async def embed() -> list[float]:
        return await provider.embed("Um texto")

    assert asyncio.run(embed()) == vector


@pytest.mark.parametrize(
    "error_type", [ProviderError, TransientProviderError, PermanentProviderError]
)
@pytest.mark.parametrize("operation", ["stream", "evaluate", "embed"])
def test_fake_raises_configured_internal_error(
    error_type: type[ProviderError], operation: str
) -> None:
    error = error_type("Falha configurada")
    provider: LLMProvider = FakeLLMProvider(error=error)

    async def invoke_provider() -> None:
        if operation == "stream":
            async for _ in provider.stream("Uma pergunta"):
                pytest.fail("The stream must raise before yielding a chunk")
        elif operation == "evaluate":
            await provider.evaluate("Uma resposta")
        else:
            await provider.embed("Um texto")

    with pytest.raises(ProviderError) as caught:
        asyncio.run(invoke_provider())

    assert caught.value is error


def test_defaults_are_usable_and_independent() -> None:
    first = FakeLLMProvider()
    second = FakeLLMProvider()
    first.chunks.append("chunk")
    first.embedding.append(1.0)
    first.evaluation.feedback = "Alterado"

    async def check_defaults() -> None:
        assert [chunk async for chunk in second.stream("prompt")] == []
        assert await second.embed("text") == []
        result = await second.evaluate("response")
        assert result.is_correct is True
        assert result.feedback == "Feedback configurado"

    asyncio.run(check_defaults())
