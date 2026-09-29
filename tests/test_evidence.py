"""证据裁剪与去重测试。"""

from langchain_core.documents import Document

from agentic_rag.evidence import deduplicate, excerpt, format_evidence


def test_excerpt_keeps_relevant_fact_near_end():
    text = "这是一段无关的开场介绍。" * 100 + "能繁母猪存栏减少3%，生猪供给出现变化。"
    result = excerpt(text, "生猪供给 能繁母猪存栏", 180)
    assert "能繁母猪存栏减少3%" in result
    assert len(result) <= 180


def test_duplicate_report_titles_do_not_count_as_independent_evidence():
    title = "2022-2027年中国猪肉行业市场供需现状及投资战略研究报告"
    documents = [Document(page_content=body, metadata={"title": title + suffix, "source_type": "web_search"})
                 for body, suffix in [("内容一", "_华经情报网"), ("内容二", "...")]]
    assert len(deduplicate(documents)) == 1


def test_actual_context_and_evidence_ids_are_bounded():
    documents = [Document(page_content="生猪" * 500, metadata={"title": "事实"}) for _ in range(4)]
    context = format_evidence(documents, "生猪", 900)
    assert len(context) <= 900
    assert all(f"[E{i}]" in context for i in range(1, 5))
