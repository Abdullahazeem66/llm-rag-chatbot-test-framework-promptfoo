import json

import pytest

from rag_app.bm25 import tokenize
from rag_app.config import RetrievalConfig
from rag_app.pipeline import RAGPipeline
from rag_app.retrieval import reciprocal_rank_fusion


def _pipeline(config, fake_embedder, fake_llm, chroma_client, **retrieval):
    cfg = config.model_copy(update={"retrieval": RetrievalConfig(**{**config.retrieval.model_dump(), **retrieval})})
    return RAGPipeline(cfg, embedder=fake_embedder, llm=fake_llm, chroma_client=chroma_client)


def test_rrf_rewards_agreement():
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    b = [{"id": "y"}, {"id": "w"}]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert [h["id"] for h in fused][:2] == ["y", "x"]
    assert all(fused[i]["score"] >= fused[i + 1]["score"] for i in range(len(fused) - 1))


def test_tokenize_keeps_prices_and_drops_stopwords():
    assert tokenize("What is the $15 plan, and E204?") == ["$15", "plan", "e204"]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_sparse_and_hybrid_modes(mode, config, fake_embedder, fake_llm, chroma_client):
    rag = _pipeline(config, fake_embedder, fake_llm, chroma_client, mode=mode)
    resp = rag.retrieve("authenticator apps security keys", top_k=2)
    assert resp.retrieved[0].doc_id == "security"
    assert [c.rank for c in resp.retrieved] == list(range(1, len(resp.retrieved) + 1))
    assert "chat" not in fake_llm.calls


def test_rewrite_only_runs_with_history(config, fake_embedder, fake_llm, chroma_client):
    fake_llm.json_responses["search_plan"] = json.dumps(
        {"standalone_query": "How long is the annual refund window?", "search_queries": []}
    )
    rag = _pipeline(config, fake_embedder, fake_llm, chroma_client, query_transform={"mode": "rewrite"})
    assert rag.retrieve("annual refunds").search_queries == ["annual refunds"]
    assert "search_plan" not in fake_llm.calls

    history = [{"role": "user", "content": "Tell me about refunds"}, {"role": "assistant", "content": "..."}]
    resp = rag.retrieve("and annual?", history=history)
    assert resp.search_queries == ["How long is the annual refund window?"]
    assert resp.usage.aux_input_tokens == 50


def test_decompose_fuses_multiple_queries(config, fake_embedder, fake_llm, chroma_client):
    fake_llm.json_responses["search_plan"] = json.dumps(
        {"standalone_query": "refunds and 2FA", "search_queries": ["monthly refund window", "two-factor authentication"]}
    )
    rag = _pipeline(config, fake_embedder, fake_llm, chroma_client, query_transform={"mode": "decompose"})
    resp = rag.retrieve("refunds and 2FA", top_k=3)
    assert resp.search_queries == ["refunds and 2FA", "monthly refund window", "two-factor authentication"]
    assert {c.doc_id for c in resp.retrieved} == {"refunds", "security"}


def test_rerank_reorders_candidates(config, fake_embedder, fake_llm, chroma_client):
    rag = _pipeline(config, fake_embedder, fake_llm, chroma_client, rerank={"enabled": True, "candidates": 3})
    fake_llm.json_responses["chunk_scores"] = json.dumps({"scores": [{"index": i, "score": 5} for i in range(3)]})
    candidates = [c.id for c in rag.retrieve("refund plans").retrieved]
    assert len(candidates) == 3

    fake_llm.json_responses["chunk_scores"] = json.dumps(
        {"scores": [{"index": 2, "score": 10}, {"index": 0, "score": 1}, {"index": 1, "score": 3}]}
    )
    resp = rag.retrieve("refund plans", top_k=2)
    assert [c.id for c in resp.retrieved] == [candidates[2], candidates[1]]
    assert [c.score for c in resp.retrieved] == [1.0, 0.3]
    assert "rerank" in resp.timings_ms


def test_reasoning_effort_reaches_each_llm_call(config, fake_embedder, fake_llm, chroma_client):
    fake_llm.json_responses["chunk_scores"] = json.dumps({"scores": [{"index": 0, "score": 9}]})
    rag = _pipeline(config, fake_embedder, fake_llm, chroma_client,
                    rerank={"enabled": True, "candidates": 3, "reasoning_effort": "none"})
    rag.retrieve("refunds")
    assert fake_llm.last_reasoning_effort == "none"
    rag.generator.config = rag.generator.config.model_copy(update={"reasoning_effort": "low"})
    rag.answer("refunds")
    assert fake_llm.last_reasoning_effort == "low"
