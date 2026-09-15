"""
shared/utils/embeddings.py
--------------------------
Reusable embedding and retrieval helpers for RAG pipelines.

Used by: learning/week06/acmecloud_rag_eval.py (and future week scripts)
"""

import pathlib
import numpy as np
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_paragraph(text: str, doc_id: str) -> list[dict]:
    """One chunk per non-empty paragraph (double-newline split)."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        {"chunk_id": f"{doc_id}::p{i}", "doc_id": doc_id, "text": p}
        for i, p in enumerate(paras)
    ]


def chunk_by_sentence(text: str, doc_id: str) -> list[dict]:
    """One chunk per sentence (regex boundary split)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    return [
        {"chunk_id": f"{doc_id}::s{i}", "doc_id": doc_id, "text": s}
        for i, s in enumerate(sentences)
    ]


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: pathlib.Path, strategy: str = "paragraph") -> list[dict]:
    """
    Load all .txt files from corpus_dir and chunk them.

    Args:
        corpus_dir: path to a project corpus folder
                    e.g. shared/data/corpus/acmecloud/
        strategy:   "paragraph" (default) | "sentence"

    Returns:
        List of chunk dicts: {chunk_id, doc_id, text}
    """
    strategies = {"paragraph": chunk_by_paragraph, "sentence": chunk_by_sentence}
    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r} — expected 'paragraph' or 'sentence'"
        )
    chunk_fn = strategies[strategy]
    chunks = []
    doc_paths = sorted(corpus_dir.glob("*.txt"))
    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_fn(text, path.name))
    print(f"[corpus] {len(doc_paths)} docs → {len(chunks)} chunks  (strategy={strategy})")
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_batch(texts: list[str], client: AsyncOpenAI, model: str) -> list[list[float]]:
    """Embed a list of texts in a single API call. Returns list of vectors."""
    resp = await client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


async def embed_corpus(chunks: list[dict], client: AsyncOpenAI, model: str) -> list[dict]:
    """
    Embed all chunks in one batch call and attach vectors in-place.

    Returns the same list with each chunk enriched by a 'vector' key.
    """
    texts   = [c["text"] for c in chunks]
    vectors = await embed_batch(texts, client, model)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks


# ---------------------------------------------------------------------------
# Similarity & retrieval
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def retrieve(
    query_vector: list[float],
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Return the top_k most similar chunks to query_vector."""
    scored = [(cosine(query_vector, c["vector"]), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# Source-recall metric
# ---------------------------------------------------------------------------

def source_recall(retrieved_chunks: list[dict], expected_sources: list[str]) -> dict:
    """
    Check whether retrieved chunks cover the expected source documents.

    Returns dict with keys: expected_sources, retrieved_docs, sources_hit, source_recall (0–1).
    """
    retrieved_docs = {c["doc_id"] for c in retrieved_chunks}
    expected       = set(expected_sources)
    hit            = expected & retrieved_docs
    return {
        "expected_sources": sorted(expected),
        "retrieved_docs":   sorted(retrieved_docs),
        "sources_hit":      sorted(hit),
        "source_recall":    len(hit) / len(expected) if expected else 1.0,
    }
