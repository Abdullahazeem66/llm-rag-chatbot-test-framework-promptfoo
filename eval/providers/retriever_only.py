"""promptfoo provider that runs retrieval only (no answer generation; query transform / rerank LLM calls still run if configured).

Output is the ranked list of chunk IDs, one per line. Full details are in metadata.
Provider config: pipeline, top_k, query_var, pinned (ignore RAG_PIPELINE override).
Test vars: query, history (optional; used by query rewriting).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import get_pipeline, get_query, parse_history, retrieval_metadata


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config") or {}
    vars_ = context.get("vars") or {}
    try:
        resp = get_pipeline(config).retrieve(
            get_query(prompt, config, context), top_k=config.get("top_k"), history=parse_history(vars_.get("history"))
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    return {
        "output": "\n".join(f"{c.rank}. {c.id} ({c.score:.3f})" for c in resp.retrieved),
        "cost": resp.usage.cost_usd,
        "metadata": retrieval_metadata(resp),
    }
