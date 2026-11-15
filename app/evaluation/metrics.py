from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """TopK 中是否至少有一个相关文档。"""
    return 1.0 if any(doc_id in relevant_doc_ids for doc_id in retrieved_doc_ids[:k]) else 0.0


def top3_recall(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    return recall_at_k(retrieved_doc_ids, relevant_doc_ids, 3)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def citation_precision(citation_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    if not citation_doc_ids:
        return 0.0
    return sum(1 for doc_id in citation_doc_ids if doc_id in relevant_doc_ids) / len(citation_doc_ids)
