import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = Field(400, gt=0)
    chunk_overlap: int = Field(50, ge=0)


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    batch_size: int = 100


class QueryTransformConfig(BaseModel):
    mode: Literal["none", "rewrite", "decompose"] = "none"
    model: str = "gpt-6-luna"
    temperature: float | None = None
    max_tokens: int = 2000
    reasoning_effort: ReasoningEffort | None = None  # None = model default (medium)
    max_queries: int = Field(3, gt=0)


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-6-luna"
    temperature: float | None = None
    max_tokens: int = 3000
    reasoning_effort: ReasoningEffort | None = None  # None = model default (medium)
    candidates: int = Field(20, gt=0)


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(5, gt=0)
    candidates: int = Field(20, gt=0)
    rrf_k: int = 60
    query_transform: QueryTransformConfig = Field(default_factory=QueryTransformConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)


class LLMConfig(BaseModel):
    model: str = "gpt-6-luna"
    temperature: float | None = None
    max_tokens: int = 2000
    reasoning_effort: ReasoningEffort | None = None


class RAGConfig(BaseModel):
    name: str = "baseline"
    corpus_dir: Path = Path("data/corpus")
    persist_dir: Path = Path("data/chroma")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @field_validator("corpus_dir", "persist_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v

    def index_key(self) -> str:
        """Configs that chunk and embed the same way share one index."""
        payload = {
            "corpus_dir": str(self.corpus_dir),
            "chunking": self.chunking.model_dump(),
            "embedding_model": self.embedding.model,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _resolve_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if not path.suffix:
        return CONFIGS_DIR / f"{path}.yaml"
    if not path.is_absolute() and not path.exists():
        return PROJECT_ROOT / path
    return path


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def _load_raw(path: Path, seen: tuple = ()) -> dict:
    if path in seen:
        raise ValueError(f"circular extends: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _load_raw(_resolve_path(parent), seen + (path,))
        base.pop("name", None)
        data = _deep_merge(base, data)
    return data


def load_config(name_or_path: str | Path = "baseline") -> RAGConfig:
    """Load a pipeline config; `extends: <name>` inherits from another config and overrides nested keys."""
    path = _resolve_path(name_or_path)
    data = _load_raw(path)
    data.setdefault("name", path.stem)
    return RAGConfig(**data)
