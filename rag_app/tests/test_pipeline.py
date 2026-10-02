import pytest

from rag_app.generation import CONTEXT_PLACEHOLDER, render_prompt
from rag_app.pipeline import RAGPipeline
from rag_app.retrieval import RetrievedChunk
from rag_app.tracing import cost_usd


@pytest.fixture
def pipeline(config, fake_embedder, fake_llm, chroma_client):
    return RAGPipeline(config, embedder=fake_embedder, llm=fake_llm, chroma_client=chroma_client)


def _chunk(id_="doc#chunk_0", text="Some text"):
    return RetrievedChunk(id=id_, text=text, score=0.9, rank=1, doc_id="doc", source="doc.md", title="Doc")


def test_render_prompt_replaces_placeholders():
    prompt = render_prompt(f"Ctx:\n{CONTEXT_PLACEHOLDER}\nQ: <<QUERY>>", "what?", [_chunk()])
    assert "[doc#chunk_0]" in prompt and "Some text" in prompt
    assert "Q: what?" in prompt
    assert CONTEXT_PLACEHOLDER not in prompt


def test_render_prompt_requires_context_placeholder():
    with pytest.raises(ValueError):
        render_prompt("no placeholder {{query}}", "q", [_chunk()])


def test_answer_response_contract(pipeline, fake_llm):
    resp = pipeline.answer("How long do I have to refund a monthly plan?")
    data = resp.model_dump()
    assert {"query", "answer", "retrieved", "rendered_prompt", "config_id", "model", "timings_ms", "usage"} <= data.keys()
    assert resp.answer.startswith("Stub answer")
    assert resp.retrieved and resp.retrieved[0].rank == 1
    assert {"id", "text", "score", "rank", "source"} <= data["retrieved"][0].keys()
    assert {"retrieve", "generate", "total"} <= resp.timings_ms.keys()
    assert resp.usage.input_tokens == 100 and resp.usage.embedding_tokens > 0
    assert resp.retrieved[0].id in fake_llm.last_messages[-1]["content"]


def test_retrieval_ranks_relevant_chunk_first(pipeline):
    resp = pipeline.retrieve("two-factor authentication authenticator apps", top_k=2)
    assert resp.retrieved[0].doc_id == "security"
    assert [c.rank for c in resp.retrieved] == [1, 2]


def test_custom_template_and_history(pipeline, fake_llm):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    resp = pipeline.answer("refunds?", prompt_template=f"ONLY THIS {CONTEXT_PLACEHOLDER}", history=history)
    assert resp.rendered_prompt.startswith("ONLY THIS")
    assert fake_llm.last_messages[:2] == history


def test_index_is_reused_and_rebuilt_on_corpus_change(config, fake_embedder, fake_llm, chroma_client, corpus_dir):
    first = RAGPipeline(config, embedder=fake_embedder, llm=fake_llm, chroma_client=chroma_client).ingest()
    assert first.rebuilt
    again = RAGPipeline(config, embedder=fake_embedder, llm=fake_llm, chroma_client=chroma_client).ingest()
    assert not again.rebuilt and again.chunks == first.chunks

    (corpus_dir / "new.md").write_text("# New\n\nBrand new document.", encoding="utf-8")
    changed = RAGPipeline(config, embedder=fake_embedder, llm=fake_llm, chroma_client=chroma_client).ingest()
    assert changed.rebuilt and changed.documents == 3


def test_cost_lookup():
    assert cost_usd("gpt-6-luna", 1_000_000, 1_000_000) == 0.6
    assert cost_usd("gpt-4.1-mini-2025-04-14", 1_000_000) == 0.4
    assert cost_usd("unknown-model", 10) is None


def test_inject_ids_put_chunks_first(pipeline, fake_llm):
    retrieved = [c.id for c in pipeline.retrieve("two-factor authentication").retrieved]
    inject = [cid for cid in pipeline.index.get_chunks(retrieved[-1:])[0]["id"].split(",")]
    resp = pipeline.answer("two-factor authentication", inject_ids=inject)
    assert resp.retrieved[0].id == inject[0] and resp.retrieved[0].metadata["injected"] is True
    assert [c.rank for c in resp.retrieved] == list(range(1, len(resp.retrieved) + 1))
    assert len({c.id for c in resp.retrieved}) == len(resp.retrieved)
    with pytest.raises(KeyError):
        pipeline.answer("x", inject_ids=["nope#chunk_9"])


def test_parse_context_round_trips_format_context():
    from rag_app.generation import format_context, parse_context

    chunks = [
        RetrievedChunk(id="a#chunk_0", text="Line one\n\n| x | y |\n|---|---|\n| 1 | 2 |", score=1, rank=1, doc_id="a", source="a.md",
                       title="A", metadata={"doc_type": "official", "last_updated": "2026-01-01"}),
        RetrievedChunk(id="custom#note_1", text="Crafted text", score=1, rank=2, doc_id="custom", source="", title=""),
    ]
    blocks = parse_context(format_context(chunks))
    assert [b.id for b in blocks] == ["a#chunk_0", "custom#note_1"]
    assert [b.text for b in blocks] == ["Line one\n\n| x | y |\n|---|---|\n| 1 | 2 |", "Crafted text"]
    assert blocks[0].header.endswith("type: official | updated: 2026-01-01")
    assert parse_context("") == [] and parse_context("  \n ") == []
    with pytest.raises(ValueError):
        parse_context("no header here")
