"""可恢复完整图、专题回读及跨运行成果复用，不依赖外部服务。"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_rag.graph import nodes
from agentic_rag.graph.build import build_graph
from agentic_rag.graph.chains import EvidenceAssessment, AnswerAssessment, DocumentAssessment
from agentic_rag.research import service
from agentic_rag.research.store import ResearchStore


class NoIndex:
    def flush(self):
        return {"synced": 0, "failed": 0, "remaining": 0}


def test_checkpoint_reopens_and_continues_without_rerunning_completed_nodes(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    doc = Document(page_content="近期供给减少。", metadata={"source_type": "news_api", "article_id": "a", "title": "供给变化"})
    noise = Document(page_content="手机发布。", metadata={"source_type": "news_api", "article_id": "noise", "title": "无关手机"})
    config = {"configurable": {"thread_id": "recover"}, "recursion_limit": 100}
    reads = []
    real_read = service.read_documents

    def read_node(state):
        def read(payload):
            reads.append(1)
            return {"claims": [{"kind": "reported_fact", "statement": "来源报道", "quote": payload["text"]}], "limitations": []}
        return real_read(state, database=database, read_fn=read, index=NoIndex())

    with (patch.object(service, "store", return_value=database),
          patch.object(service, "model_revision", return_value="test-model"),
          patch.object(service, "recall", return_value={"memory_documents": []}),
          patch.object(service, "read_documents", side_effect=read_node),
          patch.object(service, "dispatch_specialists", return_value={"next_action": "generate", "specialist_findings": []}),
          patch.object(nodes, "route", return_value={"task_type": "forecast", "selected_sources": ["news_api"], "retries": 0}),
          patch.object(nodes, "collect_sources", return_value={"documents": [doc, noise]}) as collect,
          patch.object(nodes, "get_document_grader") as grader,
          patch.object(nodes, "get_evidence_assessor") as assessor,
          patch.object(nodes, "get_generator") as generator,
          patch.object(nodes, "get_answer_reviewer") as reviewer,
          patch.object(nodes, "AnalysisCache") as cache):
        grader.return_value.batch.return_value = [
            DocumentAssessment(decision="keep", subject_match="driver", evidence_role="fact", reason="供给事实"),
            DocumentAssessment(decision="exclude", subject_match="different", evidence_role="noise", reason="无关手机")]
        assessor.return_value.invoke.return_value = EvidenceAssessment(ready=True, usable_evidence_ids=["E1"], covered_factors=["供给"], missing_factors=[], news_queries=[], web_queries=[], summary="可作条件推断")
        reviewer.return_value.invoke.return_value = AnswerAssessment(decision="accept", grounded=True, answers_question=True, needs_more_evidence=False, issues=[], revision_instructions="", news_queries=[], web_queries=[],
            concern_checks=[{"concern_id": f"C{i}", "addressed": True, "explanation": "测试替身：答案已回应"} for i in range(1, 6)])
        cache.return_value.get.return_value = None
        generator.return_value.invoke.side_effect = [RuntimeError("模拟模型故障"), "若供给继续收缩，价格可能上行 [E1]。"]
        path = str(tmp_path / "checkpoint.sqlite")
        with SqliteSaver.from_conn_string(path) as saver:
            graph = build_graph(checkpointer=saver, research=True)
            with pytest.raises(RuntimeError, match="模拟"):
                graph.invoke({"question": "预测价格", "run_id": "recover"}, config)
            assert graph.get_state(config).next == ("generate",)
        # 新建 saver 和 graph，证明不是内存中碰巧保留的状态。
        with SqliteSaver.from_conn_string(path) as saver:
            result = build_graph(checkpointer=saver, research=True).invoke(None, config)
        assert result["generation_complete"] and result["generation_grounded"]
        assert collect.call_count == 1 and len(reads) == 1
        assert grader.return_value.batch.call_count == 1
        assert len(database.tasks("recover")) == 1  # 被排除的手机没有进入阅读调度。


def test_specialist_can_request_original_once_and_reuses_across_runs(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    doc = Document(page_content="开头介绍。\n原文尾部披露，能繁母猪存栏减少。", metadata={"source_type": "news_api", "article_id": "a"})
    article = database.register(doc, 2200)
    note = Document(page_content="摘要未包含尾部细节。", metadata={"memory_version": article["version"]})
    state = {"question": "价格趋势", "task_type": "forecast", "run_id": "one", "model_revision": "v1", "answer_documents": [note], "evidence_context": "[E1] 摘要", "evidence_needs": ["能繁母猪存栏"], "reading_reports": []}
    payloads = []
    def analyze(payload):
        payloads.append(payload)
        return {"summary": "若供给收缩，价格可能上行 [E1]", "evidence_ids": ["E1"], "limitations": ["需持续跟踪"], "needs_more_evidence": False, "news_queries": [], "web_queries": [], "reread_evidence_ids": ["E1"] if "原文回读结果" not in payload["context"] else []}
    first = service.dispatch_specialists(state, database=database, analyze_fn=analyze, index=NoIndex())
    second = service.dispatch_specialists({**state, "run_id": "two"}, database=database, analyze_fn=analyze, index=NoIndex())
    assert len(payloads) == 2  # 第一次分析与一次回读；第二个研究直接复用。
    assert "能繁母猪存栏减少" in payloads[1]["context"]
    assert first["specialist_findings"] == second["specialist_findings"]
    assert "原文回读" in first["evidence_context"]
    tool_events = [event for event in database.events('one') if event.get('kind') == 'tool']
    assert [event['phase'] for event in tool_events] == ['started', 'finished']
    assert {event['tool'] for event in tool_events} == {'read_news'}
    assert len({event['tool_call_id'] for event in tool_events}) == 1
    for passage in first["specialist_findings"][0]["reread_passages"]:
        assert article["body"][passage["start"]:passage["end"]] == passage["quote"]


def test_future_prediction_gap_does_not_force_another_search(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    state = {"question": "预测", "run_id": "r", "task_type": "forecast", "model_revision": "v", "answer_documents": [Document(page_content="基准")], "evidence_context": "[E1] 基准", "evidence_needs": ["供给"]}
    result = service.dispatch_specialists(state, database=database, index=NoIndex(), analyze_fn=lambda p: {
        "summary": "条件预测 [E1]", "evidence_ids": ["E1"], "limitations": [], "needs_more_evidence": True,
        "news_queries": ["2099年产能预测"], "web_queries": [], "reread_evidence_ids": []})
    assert result["next_action"] == "generate"


def test_invalid_specialist_citation_fails_without_polluting_vector_memory(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    state = {"question": "预测", "run_id": "r", "task_type": "forecast", "model_revision": "v", "answer_documents": [Document(page_content="基准")], "evidence_context": "[E1] 基准", "evidence_needs": ["供给"]}
    result = service.dispatch_specialists(state, database=database, index=NoIndex(), analyze_fn=lambda p: {
        "summary": "错误编号 [E99]", "evidence_ids": ["E1"], "limitations": [], "needs_more_evidence": False,
        "news_queries": [], "web_queries": []})
    assert not result["specialist_findings"]
    assert database.tasks("r")[0]["status"] == "failed"
    assert not database.pending_vectors("test")


def test_same_excerpt_with_new_original_version_invalidates_specialist_cache(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    calls = []
    def analyze(payload):
        calls.append(1)
        return {"summary": "条件判断 [E1]", "evidence_ids": ["E1"], "limitations": [], "needs_more_evidence": False, "news_queries": [], "web_queries": []}
    state = {"question": "预测", "run_id": "r", "task_type": "forecast", "model_revision": "v", "answer_documents": [Document(page_content="基准", metadata={"memory_version": "v1"})], "evidence_context": "[E1] 基准", "evidence_needs": ["供给"]}
    service.dispatch_specialists(state, database=database, index=NoIndex(), analyze_fn=analyze)
    state["answer_documents"][0].metadata["memory_version"] = "v2"
    service.dispatch_specialists(state, database=database, index=NoIndex(), analyze_fn=analyze)
    assert len(calls) == 2


def test_task_budget_also_stops_assessor_supplement(tmp_path):
    from agentic_rag.config import settings
    database = ResearchStore(tmp_path / "research.sqlite")
    for i in range(settings.research_max_tasks):
        database.submit("r", str(i), "reader", {}, [], settings.research_max_tasks)
    with patch.object(service, "store", return_value=database):
        assert not nodes._can_supplement({"run_id": "r", "reading_recipe": "recipe", "retries": 0})


def test_final_cache_includes_actual_model_revision():
    assert nodes._analysis_version({"model_revision": "digest1"}, "context") != nodes._analysis_version({"model_revision": "digest2"}, "context")
