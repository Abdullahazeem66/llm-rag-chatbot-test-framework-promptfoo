from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag_app.bm25 import BM25Index
from rag_app.config import RetrievalConfig
from rag_app.embeddings import Embedder
from rag_app.index import VectorIndex
from rag_app.llm import LLMClient
from rag_app.query_transform import QueryTransformer
from rag_app.rerank import LLMReranker
from rag_app.tracing import Timings, add_costs, cost_usd


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    rank: int
    doc_id: str
    source: str
    title: str
    section: str = ""
    metadata: dict = Field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    search_queries: list[str]
    embedding_tokens: int = 0
    aux_input_tokens: int = 0
    aux_output_tokens: int = 0
    aux_cost_usd: float | None = 0.0


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    hits: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (k + rank)
            hits.setdefault(hit["id"], hit)
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [{**hits[i], "score": round(scores[i], 6)} for i in order]


class Retriever:
    def __init__(self, config: RetrievalConfig, index: VectorIndex, embedder: Embedder, llm: LLMClient):
        self.config = config
        self.index = index
        self.embedder = embedder
        self.transformer = QueryTransformer(config.query_transform, llm)
        self.reranker = LLMReranker(config.rerank, llm)
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.index.all_chunks())
        return self._bm25

    def retrieve(
        self, query: str, top_k: int | None = None, history: list[dict] | None = None, timings: Timings | None = None
    ) -> RetrievalResult:
        cfg = self.config
        timings = timings or Timings()
        top_k = top_k or cfg.top_k
        pool = max(top_k, cfg.rerank.candidates) if cfg.rerank.enabled else top_k
        aux_in = aux_out = 0
        aux_cost: float | None = 0.0

        if cfg.query_transform.mode != "none":
            with timings.track("transform"):
                plan = self.transformer.plan(query, history)
            aux_in, aux_out = plan.input_tokens, plan.output_tokens
            if plan.input_tokens:
                aux_cost = cost_usd(plan.model or cfg.query_transform.model, plan.input_tokens, plan.output_tokens)
            queries = plan.queries
        else:
            queries = [query]

        with timings.track("search"):
            ranked_lists, embedding_tokens = self._search(queries, pool)
            hits = reciprocal_rank_fusion(ranked_lists, cfg.rrf_k) if len(ranked_lists) > 1 else ranked_lists[0]
            hits = hits[:pool]

        if cfg.rerank.enabled:
            with timings.track("rerank"):
                result = self.reranker.rerank(queries[0], hits)
            hits = result.hits
            aux_in += result.input_tokens
            aux_out += result.output_tokens
            aux_cost = add_costs(aux_cost, cost_usd(result.model or cfg.rerank.model, result.input_tokens, result.output_tokens))

        chunks = [_to_chunk(hit, rank) for rank, hit in enumerate(hits[:top_k], start=1)]
        return RetrievalResult(chunks, queries, embedding_tokens, aux_in, aux_out, aux_cost)

    def _search(self, queries: list[str], pool: int) -> tuple[list[list[dict]], int]:
        mode = self.config.mode
        n = max(pool, self.config.candidates) if mode == "hybrid" else pool
        vectors, tokens = [], 0
        if mode in ("dense", "hybrid"):
            embedded = self.embedder.embed(queries)
            vectors, tokens = embedded.vectors, embedded.tokens

        ranked_lists = []
        for i, q in enumerate(queries):
            if mode == "dense":
                ranked_lists.append(self.index.query(vectors[i], n))
            elif mode == "bm25":
                ranked_lists.append(self.bm25.search(q, n))
            else:
                dense = self.index.query(vectors[i], n)
                sparse = self.bm25.search(q, n)
                ranked_lists.append(reciprocal_rank_fusion([dense, sparse], self.config.rrf_k)[:n])
        return ranked_lists, tokens


def _to_chunk(hit: dict, rank: int) -> RetrievedChunk:
    meta = hit["metadata"]
    return RetrievedChunk(
        id=hit["id"],
        text=hit["text"],
        score=hit["score"],
        rank=rank,
        doc_id=meta.get("doc_id", ""),
        source=meta.get("source", ""),
        title=meta.get("title", ""),
        section=meta.get("section", ""),
        metadata=meta,
    )
