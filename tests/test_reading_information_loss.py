"""真实 N09 回归：标题省略预计，精读不得再丢失正文限定。"""

from langchain_core.documents import Document
from agentic_rag.evidence import atomic_excerpt
from agentic_rag.research.chains import SelectedReading
import pytest
from pydantic import ValidationError


BODY = "第三方机构发布报告。报告显示，上半年用户规模预计突破6亿，较上年的1.2亿增长。行业仍面临成本压力。"


def test_duplicate_selections_do_not_push_qualifier_out_of_six_claim_limit():
    claims = [{"kind": "reported_fact", "statement": f"事实{i}", "sentence_ids": [f"S{i}"]} for i in range(1, 6)]
    claims.extend([claims[-1], {"kind": "attributed_forecast", "statement": "预计突破6亿", "sentence_ids": ["S6"]}])
    result = SelectedReading.model_validate({"claims": claims, "limitations": []})
    assert any(c.sentence_ids == ["S6"] for c in result.claims)


@pytest.mark.parametrize("ids", [None, [{"unexpected": "S1"}], [1]])
def test_bad_sentence_ids_are_validation_errors_not_dedup_crashes(ids):
    with pytest.raises(ValidationError):
        SelectedReading.model_validate({"claims": [{"kind": "reported_fact", "statement": "事实",
            "sentence_ids": ids}], "limitations": []})


def test_question_focused_original_can_recover_fact_missing_from_reading():
    doc = Document(page_content="阅读摘要漏掉了预计一句", metadata={"source_text": BODY,
        "reading_claims": [{"kind": "reported_fact", "statement": "发布了报告", "quote": "第三方机构发布报告。"}]})
    context = atomic_excerpt(doc, "用户6亿是预计还是统计结果", 500)
    assert "预计突破6亿" in context


def test_verbatim_context_does_not_split_number_and_unit():
    doc = Document(page_content=BODY, metadata={"source_text": BODY, "reading_claims": [
        {"kind": "source_excerpt", "statement": BODY, "quote": BODY}]})
    context = atomic_excerpt(doc, "预计用户规模", 65)
    assert "预计突破6亿" in context
    assert len(context) <= 65


def test_original_backfill_keeps_event_heading_when_budget_forces_excerpt():
    text = "第一阶段(1月)：金价冲顶\n国际黄金触及5595.75美元。\n国际黄金需求增加。\n国际黄金市场转强。\n" + "其他话题内容。" * 50
    doc = Document(page_content="笔记", metadata={"source_text": text})
    context = atomic_excerpt(doc, "国际黄金5595.75美元是什么时候", 50)
    assert "5595.75美元" in context
    assert "第一阶段(1月)" in context
    assert len(context) <= 50


def test_qualifier_guard_rejects_forecast_as_confirmed_but_not_qualified_answer():
    from agentic_rag.evidence import audit_forecast_status
    doc = Document(page_content=BODY, metadata={"source_text": BODY})
    assert audit_forecast_status("用户规模已突破6亿，属于统计结果[E1]。", [doc])
    assert not audit_forecast_status("用户规模预计突破6亿，不是已确认统计值[E1]。", [doc])
    assert not audit_forecast_status("报告提到上年用户1.2亿[E1]。", [doc])
    assert not audit_forecast_status("原文用户规模预计突破6亿[E1]。经算术核对，从1.2亿至6亿是5倍增长，即增长400%。", [doc])
    assert audit_forecast_status("原文预计突破6亿。现在已经突破6亿，实现5倍增长[E1]。", [doc])


def test_obvious_doubling_conflict_is_not_copied_as_an_unqualified_fact():
    from agentic_rag.evidence import audit_growth_claims
    bad = "用户规模预计突破6亿，较2025年的1.2亿实现翻倍增长[E1]。"
    assert audit_growth_claims(bad)
    assert audit_growth_claims("用户从1.2亿增长至6亿，实现翻倍增长。")
    assert not audit_growth_claims(bad + "但这些数字对应5倍，原文措辞不一致，需核实。")
    assert not audit_growth_claims("用户从1亿增长至2亿，实现翻倍增长。")
    assert not audit_growth_claims("成本从6亿元降至3亿元。")


def test_short_reading_persists_verbatim_without_a_model_call(tmp_path):
    from unittest.mock import patch
    from agentic_rag.research import service
    from agentic_rag.research.store import ResearchStore
    class NoIndex:
        def flush(self):
            return {}
    db = ResearchStore(str(tmp_path / "research.db"))
    doc = Document(page_content=BODY, metadata={"source_type": "news_api", "article_id": "short-article"})
    state = {"run_id": "short", "reading_recipe": "verbatim-test", "task_type": "factual", "documents": [doc]}
    db.start_run("short", "核对数字")
    with patch.object(service, "read_segment", side_effect=AssertionError("短原文不应重新概括")):
        result = service.read_documents(state, database=db, index=NoIndex())
    reading = result["documents"][0]
    assert reading.metadata["source_text"] == BODY
    assert reading.metadata["reading_claims"][0]["kind"] == "source_excerpt"
    assert reading.metadata["reading_status"] == "complete"
    assert result["reading_reports"][0]["covered"] == 1
