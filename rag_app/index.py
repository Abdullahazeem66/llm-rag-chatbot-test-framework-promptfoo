from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from rag_app.chunking import Chunk, chunk_corpus
from rag_app.config import RAGConfig
from rag_app.embeddings import Embedder
from rag_app.ingest import corpus_fingerprint, load_corpus


@dataclass
class IndexStats:
    collection: str
    documents: int
    chunks: int
    embedding_tokens: int
    rebuilt: bool


class VectorIndex:
    def __init__(self, config: RAGConfig, embedder: Embedder, client: chromadb.ClientAPI | None = None):
        self.config = config
        self.embedder = embedder
        self.client = client or chromadb.PersistentClient(
            path=str(config.persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self.name = f"idx_{config.index_key()}"

    def _existing(self):
        try:
            return self.client.get_collection(self.name)
        except Exception:
            return None

    def build(self, rebuild: bool = False) -> IndexStats:
        docs = load_corpus(self.config.corpus_dir)
        fingerprint = corpus_fingerprint(docs)
        existing = self._existing()
        if existing and not rebuild and (existing.metadata or {}).get("corpus_fingerprint") == fingerprint:
            return IndexStats(self.name, len(docs), existing.count(), 0, rebuilt=False)
        if existing:
            self.client.delete_collection(self.name)

        chunks = chunk_corpus(docs, self.config.chunking)
        embedded = self.embedder.embed([c.text for c in chunks])
        collection = self.client.create_collection(
            self.name,
            metadata={
                "hnsw:space": "cosine",
                "corpus_fingerprint": fingerprint,
                "embedding_model": self.embedder.model,
                "chunking": self.config.chunking.model_dump_json(),
            },
        )
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embedded.vectors,
            metadatas=[_chroma_metadata(c) for c in chunks],
        )
        return IndexStats(self.name, len(docs), len(chunks), embedded.tokens, rebuilt=True)

    def all_chunks(self) -> list[dict]:
        collection = self._existing()
        if collection is None:
            raise RuntimeError(f"Index {self.name} does not exist; run ingest first")
        res = collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def get_chunks(self, ids: list[str]) -> list[dict]:
        collection = self._existing()
        if collection is None:
            raise RuntimeError(f"Index {self.name} does not exist; run ingest first")
        res = collection.get(ids=ids, include=["documents", "metadatas"])
        found = {id_: (doc, meta) for id_, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])}
        missing = [i for i in ids if i not in found]
        if missing:
            raise KeyError(f"Chunks not in index {self.name}: {missing}")
        return [{"id": i, "text": found[i][0], "metadata": found[i][1], "score": 0.0} for i in ids]

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        collection = self._existing()
        if collection is None:
            raise RuntimeError(f"Index {self.name} does not exist; run ingest first")
        res = collection.query(
            query_embeddings=[vector], n_results=top_k, include=["documents", "metadatas", "distances"]
        )
        return [
            {"id": id_, "text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]


def _chroma_metadata(chunk: Chunk) -> dict:
    return {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
