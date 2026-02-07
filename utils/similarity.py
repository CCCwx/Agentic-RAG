"""
Text similarity for comparing Rewritten_Query_1 vs Rewritten_Query_2 (Stage 5 backtrack).
If similarity > 0.9, skip vector retrieval and use web search only.
"""
from typing import Optional


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1] (or [0,1] for non-negative)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def query_similarity(
    query1: str,
    query2: str,
    embed_model=None,
) -> float:
    """
    Embed both strings and return cosine similarity in [0, 1].
    Uses project embed_model from model.factory if embed_model not provided.
    """
    if not query1 or not query2:
        return 0.0
    if embed_model is None:
        from model.factory import embed_model as _emb
        embed_model = _emb
    e1 = embed_model.embed_query(query1)
    e2 = embed_model.embed_query(query2)
    sim = cosine_similarity(e1, e2)
    # Embeddings are typically non-negative; clamp to [0, 1] for threshold 0.9
    return max(0.0, min(1.0, sim))
