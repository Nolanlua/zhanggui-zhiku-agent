from __future__ import annotations

import json
from pathlib import Path

from app.api.schemas import UserContext
from app.evaluation.metrics import citation_precision, reciprocal_rank, recall_at_k, top3_recall
from app.rag.citation import CitationBuilder
from app.rag.hybrid import HybridRetriever


class EvaluationService:
    def __init__(self, retriever: HybridRetriever, citation_builder: CitationBuilder) -> None:
        self.retriever = retriever
        self.citation_builder = citation_builder

    async def run(self, dataset_path: str, user: UserContext, top_k: int = 5) -> dict:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / dataset_path
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        recalls: list[float] = []
        top3s: list[float] = []
        rrs: list[float] = []
        citation_scores: list[float] = []
        details: list[dict] = []
        for row in rows:
            question = row["question"]
            relevant = set(row.get("relevant_doc_ids", []))
            hits = await self.retriever.retrieve(question, user, top_k=top_k)
            _, citations = self.citation_builder.build(hits, max_chars=3000)
            retrieved_doc_ids = [hit.doc_id for hit in hits]
            citation_doc_ids = [item.doc_id for item in citations]
            recall = recall_at_k(retrieved_doc_ids, relevant, top_k)
            top3 = top3_recall(retrieved_doc_ids, relevant)
            rr = reciprocal_rank(retrieved_doc_ids, relevant)
            cp = citation_precision(citation_doc_ids, relevant)
            recalls.append(recall)
            top3s.append(top3)
            rrs.append(rr)
            citation_scores.append(cp)
            details.append(
                {
                    "question": question,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "relevant_doc_ids": sorted(relevant),
                    "recall_at_k": recall,
                    "top3_recall": top3,
                    "rr": rr,
                    "citation_precision": cp,
                }
            )
        total = len(rows)
        return {
            "total": total,
            "recall_at_k": sum(recalls) / total if total else 0.0,
            "top3_recall": sum(top3s) / total if total else 0.0,
            "mrr": sum(rrs) / total if total else 0.0,
            "citation_accuracy": sum(citation_scores) / total if total else 0.0,
            "details": details,
        }
