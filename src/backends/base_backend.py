"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMBackend(ABC):
    """Interface for LLM-like decision generators."""

    @abstractmethod
    def generate_decision(
        self, prompt: str, temperature: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate one game decision from a prompt and structured metadata."""
