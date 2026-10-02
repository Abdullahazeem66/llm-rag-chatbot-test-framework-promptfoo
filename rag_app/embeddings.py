from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    tokens: int


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> EmbeddingResult: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100, client: OpenAI | None = None):
        self.model = model
        self.batch_size = batch_size
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(max_retries=6)
        return self._client

    def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[start:start + self.batch_size])
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
            tokens += resp.usage.total_tokens
        return EmbeddingResult(vectors=vectors, tokens=tokens)
