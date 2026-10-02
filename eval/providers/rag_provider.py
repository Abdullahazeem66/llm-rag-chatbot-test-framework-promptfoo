"""promptfoo provider for the full RAG pipeline.

The rendered promptfoo prompt is used as the generation template and must contain
<<RETRIEVED_CONTEXT>>. Retrieval uses vars.query (configurable via `query_var`).

Provider config: pipeline (config name in configs/), top_k, query_var, pinned (ignore RAG_PIPELINE override).
Test vars: query, history (optional list/JSON of {role, content}), inject_ids (optional comma-separated chunk IDs
forced to the top of the context, for noisy-context tests).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import get_pipeline, get_query, parse_history, retrieval_metadata


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config") or {}
    vars_ = context.get("vars") or {}
    try:
        resp = get_pipeline(config).answer(
            get_query(prompt, config, context),
            prompt_template=prompt,
            history=parse_history(vars_.get("history")),
            top_k=config.get("top_k"),
            inject_ids=[i.strip() for i in str(vars_.get("inject_ids") or "").split(",") if i.strip()] or None,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    usage = resp.usage
    return {
        "output": resp.answer,
        "tokenUsage": {
            "total": usage.input_tokens + usage.output_tokens,
            "prompt": usage.input_tokens,
            "completion": usage.output_tokens,
        },
        "cost": usage.cost_usd,
        "metadata": {
            **retrieval_metadata(resp),
            "model": resp.model,
            "embedding_tokens": usage.embedding_tokens,
            "rendered_prompt": resp.rendered_prompt,
        },
    }
