from app.rag.hybrid import rrf_fuse
from app.rag.types import RetrievalHit


def hit(chunk_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        source="test",
    )


def test_rrf_prefers_items_in_both_lists():
    bm25 = [hit("a", 10.0), hit("b", 8.0), hit("c", 6.0)]
    vector = [hit("b", 0.9), hit("d", 0.8), hit("a", 0.7)]

    fused = rrf_fuse([bm25, vector], k=60, top_k=3)

    assert fused[0].chunk_id in ("a", "b")
    assert set(item.chunk_id for item in fused[:2]) == {"a", "b"}
    assert all(item.source == "rrf" for item in fused)


def test_rrf_empty_input():
    assert rrf_fuse([[], []]) == []
