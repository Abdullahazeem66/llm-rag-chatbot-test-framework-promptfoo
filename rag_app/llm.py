from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float | None,
        max_tokens: int,
        json_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion: ...


class OpenAIChat:
    def __init__(self, client: OpenAI | None = None):
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(max_retries=6)
        return self._client

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float | None,
        max_tokens: int,
        json_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """`json_schema` = {"name": ..., "schema": ...} enables strict structured output."""
        kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {**json_schema, "strict": True}}
        resp = self.client.chat.completions.create(**kwargs)
        return Completion(
            text=(resp.choices[0].message.content or "").strip(),
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
