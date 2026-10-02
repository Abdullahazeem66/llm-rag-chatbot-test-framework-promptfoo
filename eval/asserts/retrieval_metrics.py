"""Ranking metrics against metadata.gold_ids (and key_facts for chunking-agnostic recall).
"""
import re


def _inputs(context: dict) -> tuple[list[str], list[str], list[str], list[str], dict]:
    provider_meta = (context.get("providerResponse") or {}).get("metadata") or {}
    test_meta = (context.get("test") or {}).get("metadata") or {}
    config = context.get("config") or {}
    k = config.get("k") or len(provider_meta.get("retrieved_ids") or [])
    return (
        (provider_meta.get("retrieved_ids") or [])[:k],
        (provider_meta.get("retrieved_context") or [])[:k],
        test_meta.get("gold_ids") or [],
        test_meta.get("key_facts") or [],
        config,
    )


def _result(score: float, config: dict, reason: str) -> dict:
    return {"pass": score >= config.get("pass_at", 0), "score": round(score, 4), "reason": reason}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).strip().lower()


def recall_at_k(output, context):
    retrieved, _, gold, _, config = _inputs(context)
    if not gold:
        return _result(1.0, config, "no gold chunks")
    found = [g for g in gold if g in retrieved]
    return _result(len(found) / len(gold), config, f"found {len(found)}/{len(gold)} gold; missing {[g for g in gold if g not in retrieved]}")


def precision_at_k(output, context):
    retrieved, _, gold, _, config = _inputs(context)
    if not retrieved:
        return _result(0.0, config, "nothing retrieved")
    hits = sum(1 for r in retrieved if r in gold)
    return _result(hits / len(retrieved), config, f"{hits}/{len(retrieved)} retrieved chunks are gold")


def hit_rate(output, context):
    retrieved, _, gold, _, config = _inputs(context)
    hit = any(g in retrieved for g in gold)
    return _result(1.0 if hit else 0.0, config, "gold chunk in top-k" if hit else f"no gold in top-k: {retrieved}")


def mrr(output, context):
    retrieved, _, gold, _, config = _inputs(context)
    for rank, rid in enumerate(retrieved, start=1):
        if rid in gold:
            return _result(1 / rank, config, f"first gold at rank {rank}")
    return _result(0.0, config, "no gold in top-k")


def doc_recall(output, context):
    """Share of gold documents with at least one retrieved chunk; independent of chunk IDs."""
    retrieved, _, gold, _, config = _inputs(context)
    gold_docs = {g.split("#")[0] for g in gold}
    if not gold_docs:
        return _result(1.0, config, "no gold documents")
    retrieved_docs = {r.split("#")[0] for r in retrieved}
    missing = sorted(gold_docs - retrieved_docs)
    return _result(1 - len(missing) / len(gold_docs), config, f"missing docs: {missing}" if missing else "all gold docs retrieved")


def fact_mrr(output, context):
    """1/rank of the first retrieved chunk containing any key fact; independent of chunk IDs."""
    _, texts, _, facts, config = _inputs(context)
    if not facts:
        return _result(1.0, config, "no key facts")
    wanted = [_norm(f) for f in facts]
    for rank, text in enumerate(texts, start=1):
        if any(f in _norm(text) for f in wanted):
            return _result(1 / rank, config, f"first key fact at rank {rank}")
    return _result(0.0, config, "no key fact retrieved")


def fact_recall(output, context):
    """Share of key facts present in any retrieved chunk; independent of chunk IDs."""
    _, texts, _, facts, config = _inputs(context)
    if not facts:
        return _result(1.0, config, "no key facts")
    haystack = _norm("\n".join(texts))
    missing = [f for f in facts if _norm(f) not in haystack]
    return _result(1 - len(missing) / len(facts), config, f"missing facts: {missing}" if missing else "all key facts retrieved")
