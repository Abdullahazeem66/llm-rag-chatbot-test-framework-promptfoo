from rag_app.config import LLMConfig
from rag_app.llm import Completion, LLMClient
from rag_app.retrieval import RetrievedChunk

import re
from dataclasses import dataclass

CONTEXT_PLACEHOLDER = "<<RETRIEVED_CONTEXT>>"
QUERY_PLACEHOLDER = "<<QUERY>>"
CHUNK_SEPARATOR = "\n\n---\n\n"
_CHUNK_HEADER = re.compile(r"^\[([^\]\s]+)\](.*)$")

DEFAULT_TEMPLATE = f"""You are Bree, the customer support assistant for Brindlework, a project management platform.

Answer the user's question using ONLY the context below.
- If the context does not contain the answer, say you don't know and suggest contacting support@brindlework.example.
- If the question is ambiguous and the conversation does not make clear what it refers to (for example "it",
  "the limit", or "how much" without a plan or feature), ask one short clarifying question instead of guessing.
  You may briefly list the likely options.
- Cite the chunk IDs you used in square brackets, for example [pricing_plans#chunk_1].
- If sources conflict, prefer official documents over community content, and newer documents over older or archived ones.
- Be concise and specific.

Context:
{CONTEXT_PLACEHOLDER}

Question: {QUERY_PLACEHOLDER}
Answer:"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        meta = c.metadata
        header = f"[{c.id}] source: {c.source} | title: {c.title}"
        if meta.get("doc_type"):
            header += f" | type: {meta['doc_type']}"
        if meta.get("last_updated"):
            header += f" | updated: {meta['last_updated']}"
        blocks.append(f"{header}\n{c.text}")
    return CHUNK_SEPARATOR.join(blocks)


@dataclass
class ContextBlock:
    id: str
    header: str
    text: str


def parse_context(context: str) -> list[ContextBlock]:
    """Inverse of format_context: split a formatted context back into [id] header + text blocks."""
    blocks = []
    for raw in (b.strip("\n") for b in context.replace("\r\n", "\n").strip().split(CHUNK_SEPARATOR)):
        if not raw.strip():
            continue
        first, _, text = raw.partition("\n")
        match = _CHUNK_HEADER.match(first.strip())
        if not match:
            raise ValueError(f"Context block must start with an [id] header line: {first[:60]!r}")
        blocks.append(ContextBlock(match.group(1), first.strip(), text.strip("\n")))
    return blocks


def render_prompt(template: str, query: str, chunks: list[RetrievedChunk]) -> str:
    if CONTEXT_PLACEHOLDER not in template:
        raise ValueError(f"Prompt template must contain {CONTEXT_PLACEHOLDER}")
    return template.replace(CONTEXT_PLACEHOLDER, format_context(chunks)).replace(QUERY_PLACEHOLDER, query)


class Generator:
    def __init__(self, config: LLMConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def generate(self, prompt: str, history: list[dict] | None = None) -> Completion:
        messages = [*(history or []), {"role": "user", "content": prompt}]
        return self.llm.complete(
            messages, self.config.model, self.config.temperature, self.config.max_tokens,
            reasoning_effort=self.config.reasoning_effort,
        )
