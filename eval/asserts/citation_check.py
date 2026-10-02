"""Citation validity: every [doc#chunk_n] cited in the answer must be one of the retrieved chunks.

Assert config (optional): require_citation (default true).
"""
import re

BRACKET = re.compile(r"\[([^\]]*#chunk_\d+[^\]]*)\]")
CHUNK_ID = re.compile(r"[A-Za-z0-9_\-]+#chunk_\d+")


def extract_citations(output: str) -> set[str]:
    """Accepts [a#chunk_1] as well as grouped forms like [a#chunk_1, b#chunk_2]."""
    return {cid for group in BRACKET.findall(output) for cid in CHUNK_ID.findall(group)}


def provider_metadata(context: dict) -> dict:
    return (context.get("providerResponse") or {}).get("metadata") or {}


def get_assert(output: str, context: dict) -> dict:
    config = context.get("config") or {}
    cited = extract_citations(output)
    retrieved = set(provider_metadata(context).get("retrieved_ids") or [])

    if not cited:
        ok = not config.get("require_citation", True)
        return {"pass": ok, "score": 1.0 if ok else 0.0, "reason": "No citations in answer"}
    invalid = sorted(cited - retrieved)
    score = 1 - len(invalid) / len(cited)
    return {
        "pass": not invalid,
        "score": score,
        "reason": f"Cited {sorted(cited)}; not in retrieved set: {invalid}" if invalid else f"All citations valid: {sorted(cited)}",
    }
