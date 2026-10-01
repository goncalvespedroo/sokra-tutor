from typing import AsyncIterator

from app.ai.provider import EvaluationResult, ProviderError


class FakeLLMProvider:
    def __init__(
        self,
        chunks: list[str] | None = None,
        evaluation: EvaluationResult | None = None,
        embedding: list[float] | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else []
        self.evaluation = (
            evaluation
            if evaluation is not None
            else EvaluationResult(is_correct=True, feedback="Feedback configurado")
        )
        self.embedding = embedding if embedding is not None else []
        self.error = error

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk

    async def evaluate(self, response: str) -> "EvaluationResult":
        if self.error is not None:
            raise self.error
        return self.evaluation

    async def embed(self, text: str) -> list[float]:
        if self.error is not None:
            raise self.error
        return self.embedding
