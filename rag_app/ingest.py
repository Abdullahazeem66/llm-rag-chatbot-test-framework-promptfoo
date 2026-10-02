import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(p for p in Path(corpus_dir).rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No documents found in {corpus_dir}")
    return [load_document(p, Path(corpus_dir)) for p in paths]


def load_document(path: Path, root: Path | None = None) -> Document:
    raw = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    meta, body = split_front_matter(raw)
    body = clean_text(body)
    source = path.relative_to(root).as_posix() if root else path.name
    title = str(meta.pop("title", "")) or _first_heading(body) or path.stem
    return Document(
        doc_id=path.stem,
        title=title,
        source=source,
        text=body,
        metadata={k: str(v) for k, v in meta.items()},
    )


def split_front_matter(text: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end():]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def corpus_fingerprint(docs: list[Document]) -> str:
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode())
        h.update(doc.text.encode())
        h.update(repr(sorted(doc.metadata.items())).encode())
    return h.hexdigest()[:16]


def _first_heading(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
