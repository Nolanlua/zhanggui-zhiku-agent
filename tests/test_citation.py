from app.api.schemas import Citation
from app.rag.citation import CitationBuilder


def test_citation_validation_detects_invalid_index():
    citations = [
        Citation(index=1, doc_id="d1", chunk_id="c1", title="制度", snippet="内容"),
        Citation(index=2, doc_id="d2", chunk_id="c2", title="SOP", snippet="内容"),
    ]
    builder = CitationBuilder()
    report = builder.validate("根据制度 [1]，以及另一条 [3]。", citations)

    assert report.citation_count == 2
    assert report.valid_indexes == [1]
    assert report.invalid_indexes == [3]
    assert not report.is_valid
