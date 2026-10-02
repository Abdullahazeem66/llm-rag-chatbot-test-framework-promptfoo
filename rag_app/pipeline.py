from pydantic import BaseModel

from rag_app.config import RAGConfig, load_config
from rag_app.embeddings import Embedder, OpenAIEmbedder
from rag_app.generation import DEFAULT_TEMPLATE, Generator, render_prompt
from rag_app.index import IndexStats, VectorIndex
from rag_app.llm import LLMClient, OpenAIChat
from rag_app.retrieval import RetrievalResult, RetrievedChunk, Retriever, _to_chunk
from rag_app.tracing import Timings, Usage, add_costs, cost_usd


class RetrievalResponse(BaseModel):
    query: str
    search_queries: list[str]
    retrieved: list[RetrievedChunk]
    config_id: str
    timings_ms: dict[str, float]
    usage: Usage


class RAGResponse(RetrievalResponse):
    answer: str
    rendered_prompt: str
    model: str


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig | str = "baseline",
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        chroma_client=None,
        auto_index: bool = True,
    ):
        self.config = load_config(config) if isinstance(config, str) else config
        self.embedder = embedder or OpenAIEmbedder(self.config.embedding.model, self.config.embedding.batch_size)
        llm = llm or OpenAIChat()
        self.index = VectorIndex(self.config, self.embedder, chroma_client)
        self.retriever = Retriever(self.config.retrieval, self.index, self.embedder, llm)
        self.generator = Generator(self.config.llm, llm)
        self.auto_index = auto_index
        self._index_ready = False

    def ingest(self, rebuild: bool = False) -> IndexStats:
        stats = self.index.build(rebuild=rebuild)
        self.retriever._bm25 = None
        self._index_ready = True
        return stats

    def _ensure_index(self):
        if not self._index_ready and self.auto_index:
            self.ingest()

    def _retrieval_usage(self, result: RetrievalResult) -> Usage:
        return Usage(
            embedding_tokens=result.embedding_tokens,
            aux_input_tokens=result.aux_input_tokens,
            aux_output_tokens=result.aux_output_tokens,
            cost_usd=add_costs(cost_usd(self.embedder.model, result.embedding_tokens), result.aux_cost_usd),
        )

    def _inject(self, chunks: list[RetrievedChunk], inject_ids: list[str]) -> list[RetrievedChunk]:
        injected = [_to_chunk(hit, 0) for hit in self.index.get_chunks(inject_ids)]
        for c in injected:
            c.metadata = {**c.metadata, "injected": True}
        merged = injected + [c for c in chunks if c.id not in set(inject_ids)]
        for rank, c in enumerate(merged, start=1):
            c.rank = rank
        return merged

    def retrieve(self, query: str, top_k: int | None = None, history: list[dict] | None = None) -> RetrievalResponse:
        self._ensure_index()
        timings = Timings()
        with timings.track("retrieve"):
            result = self.retriever.retrieve(query, top_k, history, timings)
        timings.ms["total"] = timings.ms["retrieve"]
        return RetrievalResponse(
            query=query,
            search_queries=result.search_queries,
            retrieved=result.chunks,
            config_id=self.config.name,
            timings_ms=timings.ms,
            usage=self._retrieval_usage(result),
        )

    def answer(
        self,
        query: str,
        prompt_template: str | None = None,
        history: list[dict] | None = None,
        top_k: int | None = None,
        inject_ids: list[str] | None = None,
    ) -> RAGResponse:
        """`inject_ids` puts specific chunks at the top of the context (noisy-context tests)."""
        self._ensure_index()
        timings = Timings()
        with timings.track("total"):
            with timings.track("retrieve"):
                result = self.retriever.retrieve(query, top_k, history, timings)
            if inject_ids:
                result.chunks = self._inject(result.chunks, inject_ids)
            prompt = render_prompt(prompt_template or DEFAULT_TEMPLATE, query, result.chunks)
            with timings.track("generate"):
                completion = self.generator.generate(prompt, history)

        usage = self._retrieval_usage(result)
        usage.input_tokens = completion.input_tokens
        usage.output_tokens = completion.output_tokens
        usage.cost_usd = add_costs(
            usage.cost_usd, cost_usd(self.config.llm.model, completion.input_tokens, completion.output_tokens)
        )
        return RAGResponse(
            query=query,
            search_queries=result.search_queries,
            answer=completion.text,
            retrieved=result.chunks,
            rendered_prompt=prompt,
            config_id=self.config.name,
            model=completion.model,
            timings_ms=timings.ms,
            usage=usage,
        )
