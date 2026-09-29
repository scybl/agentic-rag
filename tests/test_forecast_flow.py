"""不调用外部模型，验证完整预测流程的补搜和反思闭环。"""

from unittest.mock import patch

from langchain_core.documents import Document

from agentic_rag.graph import nodes
from agentic_rag.graph.build import build_graph
from agentic_rag.graph.chains import EvidenceAssessment, AnswerAssessment
from agentic_rag.config import settings


def evidence_review(ready=True):
    return EvidenceAssessment(
        ready=ready, usable_evidence_ids=["E1"], covered_factors=["价格基准 E1"],
        missing_factors=[] if ready else ["供给"],
        news_queries=[] if ready else ["能繁母猪"], web_queries=[], summary="需要核对供给" if not ready else "可作条件分析",
    )


def answer_review(complete=True):
    return AnswerAssessment(
        decision="accept" if complete else "revise",
        grounded=True, answers_question=complete, needs_more_evidence=False,
        issues=[] if complete else ["有依据却拒绝推断"],
        revision_instructions="" if complete else "保留事实，写条件预测",
        news_queries=[], web_queries=[],
    )


def test_full_graph_rewrites_unnecessary_refusal_without_searching_again():
    document = Document(page_content="近期出栏价格上涨，需求保持稳定", metadata={"source_type": "news_api"})
    route_result = {"task_type": "forecast", "original_question": "预测2027猪肉价格",
                    "selected_sources": ["news_api"], "source_queries": {"news_api": "生猪"},
                    "evidence_needs": ["供需"], "retries": 0}
    with (patch.object(nodes, "route", return_value=route_result),
          patch.object(nodes, "collect_sources", return_value={"documents": [document]}),
          patch.object(nodes, "get_document_grader") as grader,
          patch.object(nodes, "get_evidence_assessor") as assessor,
          patch.object(nodes, "get_generator") as generator,
          patch.object(nodes, "get_answer_reviewer") as reviewer,
          patch.object(nodes, "AnalysisCache") as cache):
        cache.return_value.get.return_value = None
        grader.return_value.invoke.return_value = "yes"
        assessor.return_value.invoke.return_value = evidence_review()
        generator.return_value.invoke.side_effect = ["没有未来预测，不能回答 [E1]", "事实 [E1]。若需求持续，则可能上涨；供给增加则下行。"]
        reviewer.return_value.invoke.side_effect = [answer_review(False), answer_review()]
        result = build_graph(research=False).invoke({"question": "预测2027猪肉价格"})
    assert result["generation_complete"]
    assert result["answer_revisions"] == 1
    assert result["retries"] == 0
    assert "可能上涨" in result["generation"]
    assert "写条件预测" in generator.return_value.invoke.call_args.args[0]["revision_feedback"]
    assert grader.return_value.invoke.call_count == 1
    assert cache.return_value.put.call_count == int(settings.analysis_cache_enabled)


def test_supplement_preserves_old_evidence_and_does_not_regrade_it():
    old = Document(page_content="价格基准", metadata={"source_type": "news_api", "article_id": "old"})
    new = Document(page_content="能繁母猪减少", metadata={"source_type": "news_api", "article_id": "new"})
    state = {"question": "猪肉预测", "task_type": "forecast", "documents": [old],
             "retries": 0, "query_history": ["news_api: 生猪"]}
    with (patch.object(nodes, "get_document_grader") as grader,
          patch.object(nodes, "get_evidence_assessor") as assessor,
          patch.object(nodes, "news_api", return_value={"documents": [new]}) as news):
        grader.return_value.invoke.return_value = "yes"
        state.update(nodes.grade_documents(state))
        assessor.return_value.invoke.return_value = evidence_review(False)
        state.update(nodes.assess_evidence(state))
        assert state["next_action"] == "supplement"
        state.update(nodes.supplement_sources(state))
        state.update(nodes.grade_documents(state))
    assert {d.metadata["article_id"] for d in state["documents"]} == {"old", "new"}
    assert state["retries"] == 1
    assert grader.return_value.invoke.call_count == 2
    assert news.call_args.args[0]["news_search_plan"]["start"] == ""
    assert state["query_history"] == ["news_api: 生猪", "news_api: 能繁母猪"]


def test_insufficient_evidence_stops_supplementing_when_budget_exhausted():
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = evidence_review(False)
        result = nodes.assess_evidence({"question": "预测", "documents": [], "retries": settings.max_retries})
    assert not result["evidence_assessment"]["ready"]
    assert result["evidence_assessment"]["usable_evidence_ids"] == []
    assert result["next_action"] == "generate"
