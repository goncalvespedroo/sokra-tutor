from typing import AsyncIterator

from services.api.app.ai.provider import EvaluationResult


class FakeLLMProvider:
    def __init__(self, error):
        self.chunks = []
        self.evaluation = EvaluationResult(
            is_corect=True, feedback="Feedback configurado"
        )
        self.embeding = []
        self.error = error

    def stream(self, prompt: str) -> "AsyncIterator[str]":
        if self.error:
            raise ValueError("Simulated error")
        for chunk in self.chunks:
            yield chunk

    async def evaluate(self, response: str) -> "EvaluationResult":
        if self.error:
            raise ValueError("Simulated error")
        return self.evaluation

    async def embed(self, text: str) -> list[float]:
        if self.error:
            raise ValueError("Simulated error")
        return self.embeding
