"""真实工作流/任务/回调/SQLite 接线及旧记录兼容，不需要外部模型。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agentic_rag.graph.chains import _with_model_retry
from agentic_rag.research.scheduler import Scheduler, TaskSpec
from agentic_rag.research.store import ResearchStore
from agentic_rag.telemetry.adapters import research_trace
from agentic_rag.telemetry.exporters import BestEffortExporter, JsonExporter, read_research, write_json
from agentic_rag.telemetry.schema import Event, Span, Trace
from agentic_rag.token_usage import UsageLedger, meter_node, usage_session


class CountedModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "trace-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="PRIVATE ANSWER", usage_metadata={
            "input_tokens": 12, "output_tokens": 6, "total_tokens": 18}))])


def test_real_graph_parallel_tasks_retries_resume_and_privacy(tmp_path):
    path = tmp_path / "db.sqlite"
    store = ResearchStore(path)
    store.start_run("r", "PRIVATE QUESTION")
    chain = _with_model_retry(CountedModel())
    attempts = {}

    def handler(payload):
        key = payload["key"]
        attempts[key] = attempts.get(key, 0) + 1
        chain.invoke("PRIVATE PROMPT")
        if key.startswith("a") and attempts[key] == 1:
            raise ValueError("PRIVATE FAILURE")
        return {"ok": True}

    specs = [TaskSpec(key, "reader", {"key": key, "goal": "PRIVATE GOAL"}) for key in ("a" * 32, "b" * 32)]
    graph = StateGraph(dict)
    graph.add_node("reading", meter_node("reading", lambda state: {
        "results": Scheduler(store, "r", workers=2, attempts=2).run(specs, {"reader": handler})}))
    graph.add_edge(START, "reading")
    graph.add_edge("reading", END)
    compiled = graph.compile()
    for _ in range(2):
        with usage_session(UsageLedger(persist=lambda event: store.event("r", event))):
            compiled.invoke({})
    store.run_status("r", "completed")
    info, records = read_research(path, "r")
    trace = research_trace(info, records)
    spans = {s.span_id: s for s in trace.spans}
    calls = [s for s in spans.values() if s.kind == "llm"]
    tasks = [s for s in spans.values() if s.kind == "task"]
    assert len(calls) == 3 and len(tasks) == 3  # 恢复命中成果不会伪造第二次模型调用。
    assert sum(s.attributes["input_tokens"] for s in calls) == 36
    assert len([s for s in spans.values() if s.kind == "session"]) == 2
    assert sorted(s.status for s in tasks) == ["completed", "completed", "failed"]
    for call in calls:
        task = spans[call.parent_id]
        node = spans[task.parent_id]
        session = spans[node.parent_id]
        assert [task.kind, node.kind, session.kind] == ["task", "node", "session"]
        assert call.attributes["task_id"] == task.attributes["task_id"]
        assert call.attributes["task_attempt"] == task.attributes["attempt"]
        assert call.started_at is not None and call.finished_at is not None
    assert "PRIVATE" not in trace.model_dump_json()
    assert research_trace(info, list(reversed(records)) + records).model_dump() == trace.model_dump()


def test_legacy_missing_usage_and_pending_remain_unknown():
    records = [
        {"id": 1, "at": 10, "data": {"kind": "token_usage", "event": "call_started", "session_id": "s",
                                      "call_id": "c", "status": "pending"}},
        {"id": 2, "at": 11, "data": {"kind": "token_usage", "event": "call_finished", "session_id": "s",
                                      "call_id": "c", "status": "failed", "input_tokens": None}},
        {"id": 3, "at": 12, "data": {"kind": "token_usage", "event": "call_started", "session_id": "s",
                                      "call_id": "pending", "status": "pending"}},
    ]
    trace = research_trace({"id": "r", "status": "interrupted"}, records)
    calls = [s for s in trace.spans if s.kind == "llm"]
    assert calls[0].elapsed_seconds is None
    assert calls[0].attributes["input_tokens"] is None
    assert calls[1].finished_at is None and calls[1].status == "pending"
    assert len(trace.warnings) == 2


def test_finished_event_does_not_regress_after_delayed_started_event():
    base = {"kind": "token_usage", "session_id": "s", "call_id": "c"}
    trace = research_trace({"id": "r"}, [
        {"id": 1, "at": 2, "data": {**base, "event": "call_finished", "status": "completed", "elapsed": 1}},
        {"id": 2, "at": 1, "data": {**base, "event": "call_started", "status": "pending"}}])
    call = next(s for s in trace.spans if s.kind == "llm")
    assert call.status == "completed" and call.elapsed_seconds == 1
    assert call.attributes["status"] == "completed"


def test_candidate_and_tool_export_preserves_structure_but_not_content():
    records = [
        {"id": 1, "at": 1, "data": {"kind": "tool", "phase": "started", "tool_call_id": "t", "tool": "search",
                                      "arguments": {"token": "PRIVATE"}, "reason": "PRIVATE"}},
        {"id": 2, "at": 2, "data": {"kind": "news_candidates", "candidates": [
            {"candidate_id": "https://example.test/?token=PRIVATE", "title": "PRIVATE", "summary": "PRIVATE",
             "selected_for_reading": True, "priority": {"rank": 1, "score": -0.1,
             "components": {"semantic": -0.3, "lexical": 0.2}, "weights": {"semantic": 0.7}}}]}},
        {"id": 3, "at": 3, "data": {"kind": "tool", "phase": "failed", "tool_call_id": "t", "tool": "search",
                                      "error": "PRIVATE", "error_type": "TimeoutError", "elapsed": 2}}]
    trace = research_trace({"id": "r"}, records)
    assert "PRIVATE" not in trace.model_dump_json()
    candidate = trace.events[1].attributes["candidates"][0]
    assert candidate["score"] == -0.1 and candidate["components"]["semantic"] == -0.3
    assert candidate["selected"] is True
    assert next(s for s in trace.spans if s.kind == "tool").status == "failed"


def test_export_reader_does_not_create_or_migrate_database_and_reads_all_events(tmp_path):
    missing = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError):
        read_research(missing, "r")
    assert not missing.exists()
    path = tmp_path / "db.sqlite"
    store = ResearchStore(path)
    store.start_run("r", "q")
    for i in range(130):
        store.event("r", {"kind": "event", "count": i})
    before = path.read_bytes()
    _, records = read_research(path, "r")
    assert len(records) == 130 and path.read_bytes() == before
    with pytest.raises(ValueError, match="不存在"):
        read_research(path, "unknown")


@pytest.mark.parametrize("spans,events", [
    ([Span(span_id="root", kind="run", name="root"), Span(span_id="x", parent_id="missing", kind="node", name="x")], []),
    ([Span(span_id="root", kind="run", name="root"), Span(span_id="x", parent_id="y", kind="node", name="x"),
      Span(span_id="y", parent_id="x", kind="node", name="y")], []),
    ([Span(span_id="root", kind="run", name="root")], [Event(event_id="e", span_id="absent", name="e")]),
])
def test_invalid_trace_rejected(spans, events):
    with pytest.raises(ValidationError):
        Trace(run_id="r", spans=spans, events=events)


def test_atomic_export_failure_preserves_previous_file_and_best_effort_isolated(tmp_path, monkeypatch):
    from agentic_rag.telemetry import exporters
    path = tmp_path / "trace.json"
    write_json(path, {"old": True})
    trace = Trace(run_id="r", spans=[Span(span_id="root", kind="run", name="root")])
    def broken(*args):
        raise OSError("PRIVATE disk failure")
    monkeypatch.setattr(exporters.os, "replace", broken)
    exporter = BestEffortExporter(JsonExporter(path))
    exporter.export(trace)
    assert exporter.failures == 1 and exporter.last_error_type == "OSError"
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_independent_runs_have_no_cross_talk(tmp_path):
    store = ResearchStore(tmp_path / "db.sqlite")
    def run(key):
        store.start_run(key, "q")
        with usage_session(UsageLedger(persist=lambda e: store.event(key, e))):
            meter_node("route", lambda state: store.event(key, {"kind": "news_ranked", "candidate_count": 2}))({})
        return research_trace(*read_research(store.path, key))
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, ["a", "b"]))
    assert not {s.span_id for s in a.spans} & {s.span_id for s in b.spans}
    assert len(a.events) == len(b.events) == 5


def test_conflicting_duplicate_event_id_rejected():
    with pytest.raises(ValueError, match="冲突"):
        research_trace({"id": "r"}, [
            {"id": 1, "at": 1, "data": {"kind": "news_ranked", "count": 1}},
            {"id": 1, "at": 1, "data": {"kind": "news_ranked", "count": 2}}])


@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan")])
def test_invalid_legacy_token_counts_stay_unknown(value):
    trace = research_trace({"id": "r"}, [{"id": 1, "at": 2, "data": {
        "kind": "token_usage", "event": "call_finished", "call_id": "c", "status": "completed",
        "input_tokens": value}}])
    call = next(s for s in trace.spans if s.kind == "llm")
    assert call.attributes["input_tokens"] is None
