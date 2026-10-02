import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_app import RAGPipeline  # noqa: E402

_PIPELINES: dict[str, RAGPipeline] = {}


def pipeline_name(config: dict) -> str:
    """The RAG_PIPELINE environment variable overrides the suite's pipeline unless the provider pins it."""
    override = os.environ.get("RAG_PIPELINE")
    if override and not config.get("pinned"):
        return override
    return config.get("pipeline", "baseline")


def get_pipeline(config: dict) -> RAGPipeline:
    name = pipeline_name(config)
    if name not in _PIPELINES:
        _PIPELINES[name] = RAGPipeline(name)
    return _PIPELINES[name]


def get_query(prompt: str, config: dict, context: dict) -> str:
    query = (context.get("vars") or {}).get(config.get("query_var", "query"))
    return str(query) if query else prompt


def parse_history(value) -> list[dict] | None:
    if not value:
        return None
    history = json.loads(value) if isinstance(value, str) else value
    return [{"role": m["role"], "content": m["content"]} for m in history]


def retrieval_metadata(resp) -> dict:
    return {
        "config_id": resp.config_id,
        "search_queries": resp.search_queries,
        "retrieved_ids": [c.id for c in resp.retrieved],
        "retrieved_context": [c.text for c in resp.retrieved],
        "retrieved": [
            {"id": c.id, "rank": c.rank, "score": c.score, "source": c.source, "section": c.section}
            for c in resp.retrieved
        ],
        "timings_ms": resp.timings_ms,
        "aux_tokens": resp.usage.aux_input_tokens + resp.usage.aux_output_tokens,
    }
