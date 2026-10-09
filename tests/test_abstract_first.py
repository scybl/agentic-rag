"""摘要批筛、正文补位、无段数截断与旧检查点恢复的回归测试。"""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from agentic_rag import news_screening as screening
from agentic_rag.news_retrieval import retrieve_news
from agentic_rag.research import service
from agentic_rag.research.scheduler import Scheduler, TaskSpec, ResearchWorkPending
from agentic_rag.research.store import ResearchStore
from agentic_rag.graph.state import GraphState


class NoIndex:
    def flush(self):
        return {"synced": 0, "failed": 0, "remaining": 0}


def candidates(n):
    return [{"article_id": str(i), "candidate_id": str(i), "title": f"产业{i}", "summary": f"供给数据{i}"} for i in range(n)]


def test_all_abstracts_screened_in_batches_and_versioned_cache(tmp_path):
    db = ResearchStore(tmp_path / "test.sqlite")
    calls = []
    def invoke(payload):
        values = json.loads(payload["candidates"])
        calls.append(len(values))
        return {"decisions": [{"candidate_id": v["candidate_id"], "relation": "direct", "value": 3, "reason": "供给数据"} for v in values]}
    items = candidates(29)
    first = screening.screen_news(items, "产业趋势", database=db, invoke=invoke)
    second = screening.screen_news(items, "产业趋势", database=db, invoke=invoke)
    assert first == second and sorted(calls) == [5, 12, 12]
    assert len(first) == 29
    changed = [{**items[0], "summary": "更新的数据"}]
    screening.screen_news(changed, "产业趋势", database=db, invoke=invoke)
    screening.screen_news(items[:1], "新问题", database=db, invoke=invoke)
    screening.screen_news(items[:1], "产业趋势", database=db, invoke=invoke, signature="new-model")
    assert calls[-3:] == [1, 1, 1]


def test_missing_ids_not_cached_and_empty_abstract_not_excluded(tmp_path):
    db = ResearchStore(tmp_path / "test.sqlite")
    response = {"decisions": [{"candidate_id": "N1", "relation": "unrelated", "value": 0, "reason": "无关"}]}
    with pytest.raises(ValueError, match="编号"):
        screening.screen_news(candidates(2), "问题", database=db, invoke=lambda _: response)
    result = screening.screen_news([{**candidates(1)[0], "summary": ""}], "问题", database=db, invoke=lambda _: response)
    assert result[0]["screening"]["relation"] == "uncertain"


def test_fulltext_edits_outside_excerpt_invalidate_cache(tmp_path):
    db = ResearchStore(tmp_path / "test.sqlite")
    calls = []
    def invoke(_):
        calls.append(1)
        return {"decisions": [{"candidate_id": "N1", "relation": "direct", "value": 2, "reason": "相关"}]}
    item = {**candidates(1)[0], "content": "甲" * 10000}
    screening.screen_news([item], "问题", stage="fulltext", database=db, invoke=invoke)
    item["content"] = item["content"][:3000] + "乙" + item["content"][3001:]
    screening.screen_news([item], "问题", stage="fulltext", database=db, invoke=invoke)
    assert len(calls) == 2


def test_long_abstracts_reduce_batch_size(tmp_path, monkeypatch):
    db = ResearchStore(tmp_path / "test.sqlite")
    monkeypatch.setattr(screening, "settings", replace(screening.settings, llm_context_window=8000))
    sizes = []
    def invoke(payload):
        values = json.loads(payload["candidates"])
        sizes.append(len(values))
        return {"decisions": [{"candidate_id": v["candidate_id"], "relation": "direct", "value": 2, "reason": "相关"} for v in values]}
    values = [{**i, "summary": "有用材料" * 200} for i in candidates(13)]
    result = screening.screen_news(values, "问题", database=db, invoke=invoke)
    assert len(result) == 13 and max(sizes) < screening.BATCH_SIZE


class NewsClient:
    def __init__(self):
        self.requests, self.full = [], []
    def iter_news(self, **kwargs):
        self.requests.append(kwargs)
        yield from candidates(30)
    def article(self, key):
        self.full.append(key)
        return {"article_id": key, "content": "" if key == "0" else "重复正文" if key in {"1", "2"} else f"全文{key}"}


def test_all_abstracts_then_core_replacement_and_full_body_only():
    client = NewsClient()
    batches = []
    def screen(items, stage):
        batches.append((stage, len(items)))
        return [{**item, "screening" if stage == "abstract" else "body_screening": {
            "relation": "unrelated" if item["article_id"] == "29" or (stage == "fulltext" and item["article_id"] == "3") else "direct",
            "value": 2, "reason": "主题核对"}} for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items, *args: (items, False, None)):
        result = retrieve_news(semantic_query="产业", api_query="产业", client=client, result_k=3, summary_screener=screen)
    assert batches[0] == ("abstract", 30)
    assert len(result.items) == 3 and {i["article_id"] for i in result.items} == {"1", "4", "5"}
    assert all(i["_content_scope"] == "full" for i in result.items)
    assert set(client.full) == {"0", "1", "2", "3", "4", "5"}
    assert all(r["max_items"] is None and r["include_content"] is False and not r["start"] and not r["end"] for r in client.requests)
    assert len(result.candidates) == 30
    assert sum(i["selected_for_reading"] for i in result.candidates) == 3
    assert len([e for e in result.events if e["kind"] == "news_core_replaced"]) == 3
    assert len([e for e in result.events if e["kind"] == "news_selected"]) == 3
    assert any(e["kind"] == "news_abstract_pool" for e in result.events)


def test_explicit_candidate_contract_not_changed_by_new_screener():
    client = NewsClient()
    def unexpected(*args, **kwargs):
        pytest.fail("明确母集使用原有范围评分，不应改用开放摘要筛选")
    items = [{**i, "published_at": "2026-09-30T00:00:00Z"} for i in candidates(3)]
    client.iter_news = lambda **kwargs: iter(items)
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda values, *args: (values, False, None)):
        result = retrieve_news(semantic_query="产业", client=client, candidate_limit=3, result_k=2, summary_screener=unexpected)
    assert len(result.items) == 2


def test_scheduler_persists_more_than_old_hard_limit(tmp_path):
    db = ResearchStore(tmp_path / "test.sqlite")
    specs = [TaskSpec(str(i), "reader", {}) for i in range(205)]
    results = Scheduler(db, "run", workers=4, limit_tasks=False).run(specs, {"reader": lambda _: {"ok": True}})
    assert len(results) == len(db.tasks("run")) == 205
    assert all(t["status"] == "complete" for t in db.tasks("run"))


def test_cancelled_attempt_does_not_exhaust_recovery_budget(tmp_path):
    db = ResearchStore(tmp_path / "test.sqlite")
    db.submit("run", "key", "reader", {}, [], None)
    assert db.claim("key", "worker", lease=10, attempts=1)
    db.abandon("worker")
    assert db.task("key")["attempts"] == 0 and db.task("key")["status"] == "pending"
    assert db.claim("key", "next-worker", lease=10, attempts=1)
    # 旧执行者返回不能抢占新执行者的成果。
    with pytest.raises(RuntimeError, match="租约"):
        db.complete("key", "worker", {"bad": True})
    db.complete("key", "next-worker", {"ok": True})


def test_24_segments_all_enqueued_failure_checkpoint_resumes_only_missing(tmp_path, monkeypatch):
    db = ResearchStore(tmp_path / "test.sqlite")
    monkeypatch.setattr(service, "settings", replace(service.settings, reading_chunk_chars=200, reading_verbatim_max_chars=0))
    docs = [Document(page_content=f"第{i}篇 " + "供给变化。" * 60,
                     metadata={"source_type": "news_api", "article_id": str(i), "title": str(i)}) for i in range(12)]
    failed, calls = [True], []
    def read(payload):
        if failed[0] and payload["text"].startswith("第11篇"):
            raise ValueError("模拟该段失败")
        calls.append(payload["text"])
        return {"claims": [{"kind": "reported_fact", "statement": "供给变化", "quote": payload["text"]}], "limitations": []}
    def node(state):
        return service.read_documents(state, database=db, read_fn=read, index=NoIndex())
    workflow = StateGraph(GraphState)
    workflow.add_node("read", node)
    workflow.add_edge(START, "read")
    workflow.add_edge("read", END)
    config = {"configurable": {"thread_id": "r"}}
    path = str(tmp_path / "checkpoint.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        graph = workflow.compile(checkpointer=saver)
        with pytest.raises(ResearchWorkPending):
            graph.invoke({"question": "问题", "documents": docs, "run_id": "r", "reading_recipe": "test"}, config)
        assert graph.get_state(config).next == ("read",)
    tasks = db.tasks("r")
    assert len(tasks) == 24 and sum(t["status"] == "complete" for t in tasks) == 23
    assert db.retry_failed("r") == 1
    failed[0] = False
    with SqliteSaver.from_conn_string(path) as saver:
        graph = workflow.compile(checkpointer=saver)
        result = graph.invoke(None, config)
        assert graph.get_state(config).next == ()
    assert len(calls) == 24
    assert sum(r["covered"] for r in result["reading_reports"]) == 24
    assert all(r["status"] == "complete" for r in result["reading_reports"])


def test_legacy_end_state_recovered_under_lease_without_refetch(tmp_path, monkeypatch):
    from agentic_rag import cli
    from agentic_rag.graph.build import build_graph
    db = ResearchStore(tmp_path / "research.sqlite")
    doc = Document(page_content="保存的原文", metadata={"source_type": "news_api"})
    state = {"question": "问题", "run_id": "legacy", "research_documents": [doc], "documents": [],
        "reading_reports": [{"status": "partial", "covered": 1, "total": 2}],
        "workflow_revision": "research-workflow-v21-bounded-delivery", "model_revision": "v1",
        "reading_recipe": service.recipe("v1"), "generation": "旧的未完成提示", "generation_complete": False,
        "next_action": "finish", "execution_violations": ["未读完"]}
    monkeypatch.setattr(service, "store", lambda: db)
    monkeypatch.setattr(service, "model_revision", lambda: "v1")
    captured = []
    def trace(graph, *args, **kwargs):
        snapshot = graph.get_state({"configurable": {"thread_id": "legacy"}})
        captured.append(snapshot)
        return {**snapshot.values, "generation": "测试结果", "generation_complete": False}
    monkeypatch.setattr(cli, "run_with_trace", trace)
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_graph(checkpointer=saver)
        graph.update_state({"configurable": {"thread_id": "legacy"}}, state, as_node="evaluate_generation")
        assert not cli.ask(graph, "", run_id="legacy", resume=True)
    assert captured[0].next == ("read_documents",)
    assert captured[0].values["documents"][0].page_content == "保存的原文"
    assert captured[0].values["workflow_revision"] == service.WORKFLOW_VERSION
    assert captured[0].values["execution_violations"] == []
    assert any(e["kind"] == "reading_recovery" for e in db.events("legacy"))


def test_completed_and_unknown_workflow_not_migrated():
    from agentic_rag.research.inspection import recoverable_reading
    base = {"research_documents": ["doc"], "reading_reports": [{"status": "partial"}],
            "workflow_revision": service.WORKFLOW_VERSION, "model_revision": "v1", "reading_recipe": "r1"}
    assert recoverable_reading(SimpleNamespace(values=base, next=(), tasks=()))
    for extra in ({"generation_complete": True}, {"workflow_revision": "unknown"}, {"research_documents": []}):
        assert not recoverable_reading(SimpleNamespace(values={**base, **extra}, next=(), tasks=()))


def test_retry_reading_keeps_new_supplement_instead_of_old_material(tmp_path, monkeypatch):
    from agentic_rag import cli
    from agentic_rag.graph.build import build_graph
    db = ResearchStore(tmp_path / "research.sqlite")
    db.submit("retry", "broken", "reader", {}, [], None)
    db.claim("broken", "worker", 10, 2)
    db.fail("broken", "worker", ValueError("模拟错误"))
    old = Document(page_content="上一轮资料")
    new = Document(page_content="新补搜的资料")
    state = {"question": "问题", "documents": [new], "research_documents": [old],
             "model_revision": "v1", "reading_recipe": service.recipe("v1"), "workflow_revision": service.WORKFLOW_VERSION}
    monkeypatch.setattr(service, "store", lambda: db)
    monkeypatch.setattr(service, "model_revision", lambda: "v1")
    captured = []
    def trace(graph, *args, **kwargs):
        snapshot = graph.get_state({"configurable": {"thread_id": "retry"}})
        captured.append(snapshot)
        return {"generation": "替身", "generation_complete": False}
    monkeypatch.setattr(cli, "run_with_trace", trace)
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_graph(checkpointer=saver)
        graph.update_state({"configurable": {"thread_id": "retry"}}, state, as_node="grade_documents")
        assert not cli.ask(graph, "", run_id="retry", resume=True, retry_failed=True)
    assert captured[0].next == ("read_documents",)
    assert captured[0].values["documents"][0].page_content == "新补搜的资料"
