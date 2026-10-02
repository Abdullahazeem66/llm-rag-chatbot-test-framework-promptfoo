import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9$%.,]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "if", "in", "is",
    "it", "its", "my", "of", "on", "or", "our", "the", "to", "we", "what", "when", "which", "who", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    tokens = (t.strip(".,") for t in _TOKEN.findall(text.lower()))
    return [t for t in tokens if t and t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i]["id"]))
        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked[:top_k] if scores[i] > 0]
