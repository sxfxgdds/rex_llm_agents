"""OpenAI-compatible LLM backend for real API calls."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from src.backends.base_backend import BaseLLMBackend

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are playing a repeated Prisoner's Dilemma game. "
    "Each round you choose to either cooperate or defect. "
    "You must respond in EXACTLY this format:\n"
    "ACTION: <cooperate or defect>\n"
    "CONFIDENCE: <float between 0 and 1>\n"
    "REASON: <one sentence explanation>"
)

FALLBACK_DECISION: dict[str, Any] = {
    "action": "cooperate",
    "confidence": 0.5,
    "risk_level": 0.5,
    "reason": "Fallback: API call failed, defaulting to cooperate.",
}


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass
class OpenAIChatBackend(BaseLLMBackend):
    """LLM backend using any OpenAI-compatible API (OpenAI, DeepSeek, vLLM, Ollama, etc.)."""

    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    max_retries: int = 2
    timeout: float = 30.0
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("No API key found. Set OPENAI_API_KEY env var or pass api_key.")
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key or "dummy",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            self._client = None

    def generate_decision(
        self, prompt: str, temperature: float, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        if self._client is None:
            logger.warning("OpenAI client not initialized, using fallback decision.")
            return dict(FALLBACK_DECISION)

        api_temperature = _clip(temperature * 2.0, 0.0, 2.0)

        user_message = (
            f"{prompt}\n\n"
            f"Choose to COOPERATE or DEFECT. Respond in the exact format:\n"
            f"ACTION: <cooperate or defect>\n"
            f"CONFIDENCE: <float between 0 and 1>\n"
            f"REASON: <one sentence>"
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=api_temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=150,
                )
                text = response.choices[0].message.content or ""
                return self._parse_response(text, metadata)
            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt < self.max_retries:
                    continue

        logger.warning("All API retries exhausted, using fallback decision.")
        return dict(FALLBACK_DECISION)

    @staticmethod
    def _parse_response(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        action_match = re.search(r"ACTION:\s*(cooperate|defect)", text, re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)

        action = action_match.group(1).lower() if action_match else "cooperate"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        reason = reason_match.group(1).strip() if reason_match else text.strip()[:200]

        confidence = _clip(confidence, 0.0, 1.0)
        risk_tendency = str(metadata.get("risk_tendency", "medium"))
        risk_level = _clip(
            {"low": 0.25, "medium": 0.50, "medium_high": 0.65, "high": 0.80}.get(risk_tendency, 0.50)
            + (0.15 if action == "defect" else 0.0),
            0.0,
            1.0,
        )

        return {
            "action": action,
            "confidence": confidence,
            "risk_level": risk_level,
            "reason": reason,
        }
