"""promptfoo provider that tests generation in isolation: no retrieval, the context comes from the test.

vars.context is inserted verbatim at <<RETRIEVED_CONTEXT>> in the rendered prompt. It uses the same format the
app produces ([chunk_id] header line + text, blocks separated by ---) and may be empty.

Provider config: pipeline (LLM settings come from this config), pinned.
Test vars: query, context, history (optional).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import get_pipeline, parse_history, pipeline_name

from rag_app.generation import CONTEXT_PLACEHOLDER, parse_context
from rag_app.tracing import Timings, cost_usd


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config") or {}
    vars_ = context.get("vars") or {}
    if CONTEXT_PLACEHOLDER not in prompt:
        return {"error": f"Prompt must contain {CONTEXT_PLACEHOLDER}"}
    fixed_context = str(vars_.get("context") or "")
    try:
        blocks = parse_context(fixed_context)
        pipeline = get_pipeline(config)
        rendered = prompt.replace(CONTEXT_PLACEHOLDER, fixed_context.strip())
        timings = Timings()
        with timings.track("generate"):
            completion = pipeline.generator.generate(rendered, parse_history(vars_.get("history")))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    timings.ms["total"] = timings.ms["generate"]
    model = pipeline.config.llm.model
    return {
        "output": completion.text,
        "tokenUsage": {
            "total": completion.input_tokens + completion.output_tokens,
            "prompt": completion.input_tokens,
            "completion": completion.output_tokens,
        },
        "cost": cost_usd(model, completion.input_tokens, completion.output_tokens),
        "metadata": {
            "config_id": pipeline_name(config),
            "context_source": "fixed",
            "retrieved_ids": [b.id for b in blocks],
            "retrieved_context": [b.text for b in blocks],
            "timings_ms": timings.ms,
            "model": completion.model,
            "rendered_prompt": rendered,
        },
    }
