from rag_app.chunking import chunk_corpus, count_tokens
from rag_app.config import PROJECT_ROOT, ChunkingConfig, RAGConfig
from rag_app.ingest import corpus_fingerprint, load_corpus

REAL_CORPUS = PROJECT_ROOT / "data" / "corpus"


def test_front_matter_and_title(corpus_dir):
    docs = {d.doc_id: d for d in load_corpus(corpus_dir)}
    assert docs["refunds"].title == "Refunds"
    assert docs["refunds"].metadata == {"doc_type": "official"}
    assert not docs["refunds"].text.startswith("---")
    assert docs["security"].title == "Security"


def test_chunk_ids_are_stable_across_runs():
    cfg = ChunkingConfig()
    first = [(c.id, c.text) for c in chunk_corpus(load_corpus(REAL_CORPUS), cfg)]
    second = [(c.id, c.text) for c in chunk_corpus(load_corpus(REAL_CORPUS), cfg)]
    assert first == second
    assert len({cid for cid, _ in first}) == len(first)


def test_chunk_id_format_and_metadata():
    chunks = chunk_corpus(load_corpus(REAL_CORPUS), ChunkingConfig())
    for c in chunks:
        assert c.id == f"{c.doc_id}#chunk_{c.index}"
        assert c.metadata["source"].endswith(".md")
    pricing = [c for c in chunks if c.doc_id == "pricing_plans"]
    assert any("| Pro | $15 |" in c.text for c in pricing), "pricing table should stay in one chunk"


def test_recursive_chunks_respect_size():
    cfg = ChunkingConfig(strategy="recursive", chunk_size=120, chunk_overlap=20)
    for c in chunk_corpus(load_corpus(REAL_CORPUS), cfg):
        assert count_tokens(c.text) <= cfg.chunk_size + 5


def test_fixed_strategy_overlaps():
    cfg = ChunkingConfig(strategy="fixed", chunk_size=60, chunk_overlap=20)
    chunks = [c for c in chunk_corpus(load_corpus(REAL_CORPUS), cfg) if c.doc_id == "billing_and_refunds"]
    assert len(chunks) > 3
    assert chunks[0].text[-15:].strip()[:8] in chunks[1].text


def test_config_change_gives_new_index_key(corpus_dir):
    base = RAGConfig(corpus_dir=corpus_dir)
    assert base.index_key() == RAGConfig(corpus_dir=corpus_dir, name="other").index_key()
    assert base.index_key() != RAGConfig(corpus_dir=corpus_dir, chunking={"chunk_size": 256}).index_key()
    assert base.index_key() != RAGConfig(corpus_dir=corpus_dir, embedding={"model": "x"}).index_key()


def test_fingerprint_changes_with_content(corpus_dir):
    before = corpus_fingerprint(load_corpus(corpus_dir))
    (corpus_dir / "new.md").write_text("# New\n\nNew content.", encoding="utf-8")
    assert corpus_fingerprint(load_corpus(corpus_dir)) != before
