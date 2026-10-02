import hashlib
import math
import re

import chromadb
import pytest
from chromadb.config import Settings

from rag_app.config import RAGConfig
from rag_app.embeddings import EmbeddingResult
from rag_app.llm import Completion

DIM = 128


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder."""

    model = "fake-embedding"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return EmbeddingResult(vectors=vectors, tokens=sum(len(t.split()) for t in texts))


class FakeLLM:
    """Plain completions return a stub answer; structured calls return `json_responses[schema name]`."""

    def __init__(self):
        self.last_messages = None
        self.json_responses: dict[str, str] = {}
        self.calls: list[str] = []

    def complete(self, messages, model, temperature, max_tokens, json_schema=None, reasoning_effort=None):
        self.last_reasoning_effort = reasoning_effort
        if json_schema:
            self.calls.append(json_schema["name"])
            return Completion(text=self.json_responses[json_schema["name"]], model=model, input_tokens=50, output_tokens=5)
        self.calls.append("chat")
        self.last_messages = messages
        return Completion(text="Stub answer [doc#chunk_0].", model=model, input_tokens=100, output_tokens=10)


SAMPLE_DOCS = {
    "refunds.md": "---\ntitle: Refunds\ndoc_type: official\n---\n\n# Refunds\n\n## Monthly\n\n"
                  "Monthly plans are refundable within 14 days.\n\n## Annual\n\nAnnual plans are refundable within 30 days.",
    "security.md": "# Security\n\n## 2FA\n\nTwo-factor authentication uses authenticator apps and security keys.",
}


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in SAMPLE_DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


@pytest.fixture
def config(tmp_path, corpus_dir):
    return RAGConfig(name="test", corpus_dir=corpus_dir, persist_dir=tmp_path / "chroma")


@pytest.fixture
def chroma_client():
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False, allow_reset=True))
    yield client
    client.reset()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()
