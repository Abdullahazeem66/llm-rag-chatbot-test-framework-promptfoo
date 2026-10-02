import json
from dataclasses import dataclass

from rag_app.config import RerankConfig
from rag_app.llm import LLMClient

SYSTEM_PROMPT = """You rank Brindlework help-center chunks by how useful they are for answering a customer's question.
Score every chunk from 0 to 10:
10 = directly answers the question or contains a fact needed for the answer
5 = on the right topic but does not contain a needed fact
0 = irrelevant
For multi-part questions, a chunk that answers any one part deserves a high score.
Official current documents are more useful than archived or community documents on the same topic."""

SCHEMA = {
    "name": "chunk_scores",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}, "score": {"type": "integer"}},
                    "required": ["index", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

MAX_CHARS = 1500


@dataclass
class RerankResult:
    hits: list[dict]
    input_tokens: int
    output_tokens: int
    model: str


class LLMReranker:
    def __init__(self, config: RerankConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def rerank(self, query: str, hits: list[dict]) -> RerankResult:
        if not hits:
            return RerankResult([], 0, 0, self.config.model)
        listing = "\n\n".join(
            f"[{i}] ({h['metadata'].get('title', '')}, {h['metadata'].get('doc_type', 'official')})\n{h['text'][:MAX_CHARS]}"
            for i, h in enumerate(hits)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nChunks:\n{listing}"},
        ]
        completion = self.llm.complete(
            messages, self.config.model, self.config.temperature, self.config.max_tokens,
            json_schema=SCHEMA, reasoning_effort=self.config.reasoning_effort,
        )
        scores = {s["index"]: s["score"] for s in json.loads(completion.text)["scores"]}
        order = sorted(range(len(hits)), key=lambda i: (-scores.get(i, 0), i))
        reranked = [{**hits[i], "score": round(scores.get(i, 0) / 10, 4)} for i in order]
        return RerankResult(reranked, completion.input_tokens, completion.output_tokens, completion.model)
