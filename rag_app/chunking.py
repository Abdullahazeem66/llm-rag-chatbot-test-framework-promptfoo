import re
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

from rag_app.config import ChunkingConfig
from rag_app.ingest import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str
    section: str = ""
    metadata: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_corpus(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, cfg)]


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if cfg.strategy == "fixed":
        pieces = [("", t) for t in split_fixed(doc.text, cfg.chunk_size, cfg.chunk_overlap)]
    else:
        pieces = split_recursive(doc.text, cfg.chunk_size, cfg.chunk_overlap)
    return [
        Chunk(
            id=f"{doc.doc_id}#chunk_{i}",
            doc_id=doc.doc_id,
            index=i,
            text=text,
            section=section,
            metadata={
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "section": section,
                "chunk_index": i,
                **doc.metadata,
            },
        )
        for i, (section, text) in enumerate(pieces)
    ]


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    tokens = _encoding().encode(text)
    step = size - overlap
    out = []
    for start in range(0, len(tokens), step):
        out.append(_encoding().decode(tokens[start:start + size]).strip())
        if start + size >= len(tokens):
            break
    return [t for t in out if t]


def split_recursive(text: str, size: int, overlap: int) -> list[tuple[str, str]]:
    """Split on markdown sections, then pack paragraphs up to `size` tokens.

    Each chunk starts with its section heading so it stays meaningful on its own.
    Paragraphs larger than `size` fall back to fixed windows with `overlap`.
    """
    out: list[tuple[str, str]] = []
    for section, heading, paragraphs in _sections(text):
        prefix = f"{heading}\n\n" if heading else ""
        budget = size - count_tokens(prefix)
        current: list[str] = []
        current_tokens = 0

        def flush():
            nonlocal current, current_tokens
            if current:
                out.append((section, prefix + "\n\n".join(current)))
            current, current_tokens = [], 0

        for para in paragraphs:
            n = count_tokens(para)
            if n > budget:
                flush()
                out.extend((section, prefix + piece) for piece in split_fixed(para, budget, overlap))
                continue
            if current_tokens + n > budget:
                flush()
            current.append(para)
            current_tokens += n
        flush()
    return out


def _sections(text: str) -> list[tuple[str, str, list[str]]]:
    sections: list[tuple[str, str, list[str]]] = []
    path: list[str] = []
    heading_line = ""
    lines: list[str] = []

    def close():
        body = "\n".join(lines).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if paragraphs:
            sections.append((" > ".join(path), heading_line, paragraphs))

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            close()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
            heading_line = line.strip()
            lines = []
        else:
            lines.append(line)
    close()
    return sections
