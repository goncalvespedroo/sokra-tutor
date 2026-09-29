from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class EvaluationResult:
    is_correct: bool
    feedback: str


class LLMProvider(Protocol):
    def stream(self, prompt: str) -> "AsyncIterator[str]": ...

    async def evaluate(self, response: str) -> "EvaluationResult": ...

    async def embed(self, text: str) -> list[float]: ...
