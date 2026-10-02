import json
from dataclasses import dataclass, field

from rag_app.config import QueryTransformConfig
from rag_app.llm import LLMClient

SYSTEM_PROMPT = """You turn a customer's latest message into search queries for the Brindlework help-center search engine.

Return:
- standalone_query: the latest message rewritten as a self-contained question. Resolve pronouns and references
  ("it", "that plan", "and Pro?") using the conversation history. If it is already self-contained, repeat it unchanged.
- search_queries: {search_instruction}

Do not answer the question."""

DECOMPOSE = ("up to {n} short search queries, one per distinct piece of information needed to answer. "
             "A single-fact question needs exactly one query. For questions that depend on another fact "
             "(e.g. 'the plan that includes X'), add a query for that fact.")
REWRITE = "an empty list."

SCHEMA = {
    "name": "search_plan",
    "schema": {
        "type": "object",
        "properties": {
            "standalone_query": {"type": "string"},
            "search_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["standalone_query", "search_queries"],
        "additionalProperties": False,
    },
}


@dataclass
class SearchPlan:
    queries: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)


class QueryTransformer:
    def __init__(self, config: QueryTransformConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def plan(self, query: str, history: list[dict] | None = None) -> SearchPlan:
        mode = self.config.mode
        if mode == "none" or (mode == "rewrite" and not history):
            return SearchPlan(queries=[query])

        instruction = DECOMPOSE.format(n=self.config.max_queries) if mode == "decompose" else REWRITE
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history or []) or "(none)"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(search_instruction=instruction)},
            {"role": "user", "content": f"Conversation history:\n{convo}\n\nLatest message: {query}"},
        ]
        completion = self.llm.complete(
            messages, self.config.model, self.config.temperature, self.config.max_tokens,
            json_schema=SCHEMA, reasoning_effort=self.config.reasoning_effort,
        )
        data = json.loads(completion.text)
        queries = [data["standalone_query"].strip() or query]
        if mode == "decompose":
            queries += data["search_queries"][: self.config.max_queries]
        unique = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        return SearchPlan(unique, completion.input_tokens, completion.output_tokens, completion.model, data)
