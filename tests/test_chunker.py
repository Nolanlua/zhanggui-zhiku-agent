from app.rag.chunker import ChunkerConfig, StructureAwareChunker
from app.rag.types import ParsedBlock, ParsedDocument


def test_chunker_preserves_section_and_page():
    doc = ParsedDocument(
        doc_id="doc_1",
        title="陈列制度",
        source_type="docx",
        blocks=[
            ParsedBlock(text="第一章 新品陈列\n" + "新品需要放在黄金位置。" * 40, page=2, section="第一章 新品陈列"),
            ParsedBlock(text="| SKU | 位置 |\n| --- | --- |\n| A | 端架 |", page=3, section="陈列表", kind="table"),
        ],
    )
    chunker = StructureAwareChunker(ChunkerConfig(max_chars=300, overlap_chars=40))
    chunks = chunker.chunk_document(doc, tenant_id="tenant_demo")

    assert len(chunks) >= 2
    assert all(chunk.tenant_id == "tenant_demo" for chunk in chunks)
    assert chunks[0].section == "第一章 新品陈列"
    assert chunks[0].page == 2
    assert any(chunk.text.startswith("| SKU |") for chunk in chunks)
