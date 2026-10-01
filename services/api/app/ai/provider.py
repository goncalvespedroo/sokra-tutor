from dataclasses import dataclass
from typing import AsyncIterator, Protocol


class ProviderError(Exception):
    """Base error for failures in the internal AI provider layer."""


class TransientProviderError(ProviderError):
    """Provider failure that may succeed on retry."""


class PermanentProviderError(ProviderError):
    """Provider failure that requires a change before retrying."""


@dataclass
class EvaluationResult:
    is_correct: bool
    feedback: str


class LLMProvider(Protocol):
    def stream(self, prompt: str) -> "AsyncIterator[str]": ...

    async def evaluate(self, response: str) -> "EvaluationResult": ...

    async def embed(self, text: str) -> list[float]: ...
