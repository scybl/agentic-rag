"""研究 c4676d2c 暴露的问题：先筛后读、补搜流上下文、风险核验与总计。"""

import importlib
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END

from agentic_rag.config import settings
from agentic_rag.graph import nodes
from agentic_rag.graph.chains import DocumentAssessment
from agentic_rag.evidence_audit import build_concerns, enforce_concern_checks
from agentic_rag.news_retrieval import NewsRetrievalResult
from agentic_rag.research.chains import SelectedReading, numbered_sentences, resolve_selection
from agentic_rag.research.chains import read_segment
from agentic_rag.research.memory import MemoryIndex
from agentic_rag.research.scheduler import Scheduler, TaskSpec
from agentic_rag.research.store import ResearchStore
from agentic_rag.token_usage import UsageLedger, print_usage_summary
from agentic_rag.tools import search_news
from agentic_rag.tools.contracts import ToolContext
from agentic_rag.tools.execution import execute_tool
from agentic_rag.evidence import format_evidence


def test_supplement_stream_preserves_checkpoint_config_after_sqlite_resume(tmp_path):
    news_module = importlib.import_module("agentic_rag.tools.news")
    def retrieve(**kwargs):
        kwargs["on_event"]({"kind": "news_request", "query": "苹果", "page": 1})
        return NewsRetrievalResult(items=[])
    def supplement(state):
        execute_tool(search_news, {"semantic_query": "苹果期货", "queries": ["苹果"]},
                     context=ToolContext(emit=get_stream_writer(), caller="测试恢复节点",
                                         reason="验证恢复后工具事件仍能写入自定义流"))
        return {"done": True}
    flow = StateGraph(dict)
    flow.add_node("supplement", supplement)
    flow.add_edge(START, "supplement")
    flow.add_edge("supplement", END)
    path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "repro"}}
    with SqliteSaver.from_conn_string(path) as saver:
        list(flow.compile(checkpointer=saver, interrupt_before=["supplement"]).stream({}, config))
    with SqliteSaver.from_conn_string(path) as saver, patch.object(news_module, "retrieve_news", side_effect=retrieve):
        events = list(flow.compile(checkpointer=saver).stream(None, config, stream_mode=["updates", "custom"]))
    assert any(kind == "custom" and event.get("kind") == "news_request" for kind, event in events)
    assert any(kind == "updates" and event.get("supplement", {}).get("done") for kind, event in events)


def test_grading_keeps_report_excludes_phone_and_audits_reasons():
    docs = [Document(page_content="苹果产量增长4.3%，消费偏弱，预计收购价下跌。", metadata={"title": "苹果产销报告"}),
            Document(page_content="iPhone手机发布", metadata={"title": "苹果新品"})]
    events = []
    with patch.object(nodes, "get_document_grader") as grader, patch.object(nodes, "get_stream_writer", return_value=events.append):
        grader.return_value.batch.return_value = [
            DocumentAssessment(decision="keep", subject_match="same", evidence_role="fact", reason="增产和消费事实，虽未给2027预测但相关"),
            DocumentAssessment(decision="exclude", subject_match="different", evidence_role="noise", reason="手机公司不是农产品")]
        result = nodes.grade_documents({"question": "预测2027苹果期货", "documents": docs,
            "subject_scope": "农业水果，不是手机", "evidence_needs": ["产量"]})
        assert grader.return_value.batch.call_args.args[0][0]["subject_scope"] == "农业水果，不是手机"
        configs = grader.return_value.batch.call_args.kwargs["config"]
        assert all(config["max_concurrency"] == settings.llm_concurrency for config in configs)
        assert grader.return_value.batch.call_args.kwargs["return_exceptions"] is True
    assert result["documents"] == [docs[0]]
    assert [event["decision"] for event in events] == ["keep", "exclude"]
    assert all(event["reason"] for event in events)


@pytest.mark.parametrize("failure", [OutputParserException("bad json"), "VERDICT missing"])
def test_parser_failure_is_uncertain_not_silent_exclusion(failure):
    doc = Document(page_content="农业事实")
    with patch.object(nodes, "get_document_grader") as grader:
        if isinstance(failure, Exception):
            grader.return_value.batch.return_value = [failure]
        else:
            grader.return_value.batch.return_value = [failure]
        result = nodes.grade_documents({"question": "预测", "documents": [doc]})
    assert result["documents"] == [doc]
    assert doc.metadata["relevance_decision"] == "uncertain"


def test_conflicting_relevance_flags_do_not_delete_substantive_evidence():
    with patch.object(nodes, "get_document_grader") as grader:
        grader.return_value.batch.return_value = [DocumentAssessment(decision="exclude", subject_match="same", evidence_role="fact", reason="没有目标年份预测")]
        result = nodes.grade_documents({"question": "预测", "documents": [Document(page_content="供给事实")]})
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["relevance_decision"] == "uncertain"


def test_nearest_memory_without_subject_match_is_not_returned():
    class Store:
        def lexical_readings(self, *args): return ["pig"]
        def current_readings(self, *args):
            return [{"id": "pig", "body": "生猪存栏减少", "metadata": {"source_type": "news_api"}, "version_id": "v"}]
    index = MemoryIndex(Store())
    with patch.object(index, "collection", side_effect=RuntimeError("offline")):
        docs, info = index.recall("预测苹果期货", "recipe", subject_terms=["苹果", "红富士"])
    assert docs == [] and info["subject_filtered"] == 1


def test_news_ranking_gets_entity_scope_without_changing_literal_filters():
    with patch.object(nodes, "_search_source", return_value={}) as search:
        nodes.news_api({"question": "苹果期货预测", "subject_scope": "农产品水果",
                        "evidence_needs": ["产量库存"], "news_search_plan": {
                            "query": "苹果", "queries": ["苹果"], "start": "", "end": "", "section": ""}})
    arguments = search.call_args.args[1]
    assert "农产品水果" in arguments["semantic_query"] and "产量库存" in arguments["semantic_query"]
    assert arguments["queries"] == ["苹果"] and arguments["start"] == arguments["end"] == ""


def test_too_many_reading_claims_preserves_bounded_result_without_retry():
    text = "苹果增产。"
    sentences, _ = numbered_sentences(text)
    result = resolve_selection({"claims": [{"kind": "reported_fact", "statement": "增产", "sentence_ids": ["S1"]}] * 9, "limitations": []}, text, sentences)
    assert len(result["claims"]) == 6 and "仅保留前6项" in result["limitations"][0]
    assert SelectedReading.model_json_schema()["properties"]["claims"]["maxItems"] == 6


def test_forecast_requires_time_scope_counterevidence_baseline_and_mechanism_checks():
    concerns = build_concerns([], [Document(page_content="历史数据")], task_type="forecast", question="苹果期货")
    assert {c["category"] for c in concerns} == {"time", "scope", "counterevidence", "baseline", "mechanism"}
    review = {"decision": "accept", "grounded": True, "issues": [], "concern_checks": []}
    enforce_concern_checks(review, concerns)
    assert review["decision"] == "revise" and not review["grounded"]
    review.update(decision="accept", grounded=True, issues=[], concern_checks=[
        {"concern_id": c["id"], "addressed": True, "explanation": "明确陈述不确定性"} for c in concerns])
    enforce_concern_checks(review, concerns)
    assert review["decision"] == "accept"
    mechanism_check = review["concern_checks"][-1]
    mechanism_check.update(addressed=False, explanation="将个体保险套保推成市场价格稳定，缺乏传导依据。")
    assert enforce_concern_checks(review, concerns)["decision"] == "revise"
    mechanism_check.update(addressed=True, explanation="已删除缺乏传导依据的价格稳定结论。")
    review.update(decision="accept", grounded=True, issues=[])
    review["concern_checks"].append(review["concern_checks"][0])
    assert enforce_concern_checks(review, concerns)["decision"] == "revise"
    review["concern_checks"] = review["concern_checks"][:-1]
    review.update(decision="accept", grounded=True, issues=[])
    review["concern_checks"][0]["explanation"] = "错误。把2025年的预测当成2027年事实。"
    assert enforce_concern_checks(review, concerns)["decision"] == "revise"


def test_search_date_hint_is_unverified_not_publication_fact():
    module = importlib.import_module("agentic_rag.tools.web_search")
    hint = module.date_hint({"body": "February 18, 2025 - 预计产量同比增长5.5%"})
    doc = Document(page_content="预计产量增长", metadata={"date_hint": hint})
    context = format_evidence([doc], "产量", 1000)
    assert "February 18, 2025" in context and "未核实" in context


def test_final_total_includes_session_warmup_but_not_replayed_history(capsys):
    session = UsageLedger()
    previous = UsageLedger(observer=session.observe)
    cid = previous.begin_call()
    previous.finish_call(cid, metadata={"prompt_eval_count": 10, "eval_count": 2})
    history = list(previous.calls.values())
    current = UsageLedger(history=history, observer=session.observe)
    cid = current.begin_call()
    current.finish_call(cid, metadata={"prompt_eval_count": 20, "eval_count": 5})
    print_usage_summary(current, session_ledger=session)
    output = capsys.readouterr().out.split("Token 最终合计")[-1]
    assert "总计 25 token" in output and "总计 37 token" in output
    assert session.report()["cumulative"]["calls"] == 2
    assert current.report()["current"]["reasoning_tokens"] is None


def test_truncated_json_is_rejected_even_if_parser_salvages_it_and_not_retried(tmp_path):
    from agentic_rag.graph import chains
    message = AIMessage(content='{"claims":[]}', response_metadata={"done_reason": "length"})
    calls = []
    with patch.object(chains, "get_llm") as llm:
        llm.return_value.with_structured_output.return_value = RunnableLambda(lambda value: {
            "raw": message, "parsed": {"claims": []}, "parsing_error": None})
        chain = chains.checked_structured(SelectedReading)
    def handler(payload):
        calls.append(1)
        return chain.invoke(payload)
    database = ResearchStore(tmp_path / "db.sqlite")
    result = Scheduler(database, "truncation", attempts=3).run([TaskSpec("truncated", "reader", {})], {"reader": handler})
    assert result["truncated"] is None and len(calls) == 1
    assert database.task("truncated")["status"] == "failed"
    Scheduler(database, "truncation", attempts=3).run([TaskSpec("truncated", "reader", {})], {"reader": handler})
    assert len(calls) == 1  # 下一补搜批次也不重复消耗。
    assert not database.task("truncated")["retryable"]
    assert not database.claim("truncated", "another-process", 60, 10)
    database.retry_failed("truncation")
    assert database.task("truncated")["retryable"]
    ledger = UsageLedger()
    cid = ledger.begin_call()
    ledger.finish_call(cid, metadata={"done_reason": "length", "prompt_eval_count": 100, "eval_count": 4096})
    assert ledger.report()["current"]["total_tokens"] == 4196
    assert ledger.report()["current"]["failed"] == 1


def test_structured_output_is_rejected_with_reason_then_model_corrects_it(monkeypatch):
    from dataclasses import replace
    from pydantic import Field
    from agentic_rag.graph import chains

    class Expected(chains.StructuredOutput):
        count: int = Field(ge=1)
        label: str

    calls = []

    def answer(messages):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "raw": AIMessage(content='{"count":"bad","extra":true}'),
                "parsed": None,
                "parsing_error": "count 必须是整数；label 缺失；extra 不允许",
            }
        return {
            "raw": AIMessage(content='{"count":2,"label":"ok"}'),
            "parsed": Expected(count=2, label="ok"),
            "parsing_error": None,
        }

    class FakeModel:
        def with_structured_output(self, schema, include_raw=False):
            assert schema is Expected and include_raw
            return RunnableLambda(answer)

    monkeypatch.setattr(chains, "get_llm", lambda: FakeModel())
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_max_attempts=2))
    result = chains.checked_structured(Expected).invoke([HumanMessage(content="生成结果")])
    assert result == Expected(count=2, label="ok")
    assert len(calls) == 2
    assert any(isinstance(message, AIMessage) and "extra" in str(message.content)
               for message in calls[1])
    feedback = next(message.content for message in calls[1] if isinstance(message, HumanMessage)
                    and "未通过" in str(message.content))
    assert all(reason in feedback for reason in ("count", "label", "extra", "重新完成"))


def test_plain_text_output_is_also_validated_and_retried(monkeypatch):
    from dataclasses import replace
    from agentic_rag.graph import chains

    calls = []

    def answer(messages):
        calls.append(messages)
        return AIMessage(content="" if len(calls) == 1 else "合格文本")

    monkeypatch.setattr(chains, "get_llm", lambda: RunnableLambda(answer))
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_max_attempts=2))
    result = chains.checked_text("FinalAnswer").invoke([HumanMessage(content="回答问题")])
    assert result == "合格文本" and len(calls) == 2
    feedback = next(message.content for message in calls[1] if isinstance(message, HumanMessage)
                    and "未通过" in str(message.content))
    assert "空内容" in feedback and "重新完成" in feedback


def test_truncated_text_dynamically_increases_single_request_budget(monkeypatch):
    from dataclasses import replace
    from agentic_rag.graph import chains

    calls, states, events = [], [], []

    def answer(messages):
        calls.append(messages)
        if len(calls) == 1:
            return AIMessage(
                content="半截答案", response_metadata={"done_reason": "length"},
                usage_metadata={"input_tokens": 3670, "output_tokens": 5000,
                                "total_tokens": 8670},
            )
        return AIMessage(content="完整答案", response_metadata={"done_reason": "stop"})

    fake = RunnableLambda(answer)
    configured = replace(
        chains.settings, llm_max_attempts=2, llm_adaptive_max_attempts=2,
        llm_max_output_tokens=8192, llm_context_window=16384,
        llm_request_timeout=300, llm_adaptive_max_output_tokens=12288,
        llm_adaptive_max_context_window=16384,
        llm_adaptive_max_request_timeout=600, llm_reasoning=True,
    )
    monkeypatch.setattr(chains, "settings", configured)
    monkeypatch.setattr(chains, "get_llm", lambda: fake)

    def runtime(state, base=None):
        states.append(state)
        return fake

    monkeypatch.setattr(chains, "_runtime_llm", runtime)
    chain = chains.checked_text("ResearchAnswer")
    result = chain.invoke([HumanMessage(content="生成研究答案")], config={
        "configurable": {"tool_context": ToolContext(
            emit=events.append, caller="测试", reason="观察动态调整")}
    })
    assert result == "完整答案" and len(calls) == 2
    assert states[0].num_predict == 5000 and states[1].num_predict == 7168
    adjustment = next(event for event in events if event.get("tool") == "plan_model_retry"
                      and event.get("phase") == "finished")
    assert adjustment["action"] == "increase_output_budget"
    assert adjustment["adjustments"]["num_predict"] == 7168


def test_stage_policy_disables_reasoning_for_direct_outputs(monkeypatch):
    from dataclasses import replace
    from agentic_rag.graph import chains

    monkeypatch.setattr(chains, "settings", replace(
        chains.settings, llm_reasoning=True, llm_max_output_tokens=8192,
    ))
    assert chains._initial_attempt_state("DocumentAssessment").num_predict == 600
    assert not chains._initial_attempt_state("DocumentAssessment").reasoning
    assert chains._initial_attempt_state("SelectedReading").num_predict == 2200
    assert not chains._initial_attempt_state("SelectedReading").reasoning
    assert chains._initial_attempt_state("EvidenceAssessment").num_predict == 4800
    assert chains._initial_attempt_state("EvidenceAssessment").reasoning
    assert chains._initial_attempt_state("ResearchAnswer").num_predict == 5000
    assert chains._initial_attempt_state("ResearchAnswer").reasoning


def test_timeout_retry_disables_reasoning_before_increasing_time(monkeypatch):
    from dataclasses import replace
    import httpx
    from agentic_rag.graph import chains

    calls, states = [], []

    def answer(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return AIMessage(content="恢复成功")

    fake = RunnableLambda(answer)
    monkeypatch.setattr(chains, "settings", replace(
        chains.settings, llm_max_attempts=2, llm_adaptive_max_attempts=2,
        llm_reasoning=True, llm_request_timeout=300,
        llm_adaptive_max_request_timeout=600,
    ))
    monkeypatch.setattr(chains, "get_llm", lambda: fake)

    def runtime(state, base=None):
        states.append(state)
        return fake

    monkeypatch.setattr(chains, "_runtime_llm", runtime)
    assert chains.checked_text("ResearchAnswer").invoke("回答") == "恢复成功"
    assert len(calls) == 2 and states[0].reasoning and not states[1].reasoning
    assert states[1].timeout_seconds == 300


def test_structured_models_forbid_unknown_fields():
    from pydantic import ValidationError
    from agentic_rag.graph.chains import DocumentAssessment

    with pytest.raises(ValidationError, match="extra_forbidden"):
        DocumentAssessment.model_validate({
            "decision": "keep", "subject_match": "same", "evidence_role": "fact",
            "reason": "相关事实", "invented": "不允许",
        })


def test_truncated_reading_splits_once_preserving_exact_original_quotes():
    from agentic_rag.graph.chains import ModelOutputTruncatedError
    calls, splits = [], []
    original = "甲" * 349 + "。" + "乙" * 349 + "。"
    def invoke(payload):
        calls.append(payload["text"])
        if len(calls) == 1:
            raise ModelOutputTruncatedError("length")
        return {"claims": [{"kind": "reported_fact", "statement": "原文事实", "sentence_ids": ["S1"]}], "limitations": []}
    result = read_segment({"goal": "阅读", "header": {}, "text": original}, invoke=invoke, on_split=splits.append)
    assert len(calls) == 3 and len(splits) == 1 and sum(splits[0]) == len(original)
    assert len(result["claims"]) == 2 and all(c["quote"] in original for c in result["claims"])
    assert result["limitations"]
    calls.clear()
    def always_fail(payload):
        calls.append(1)
        raise ModelOutputTruncatedError("length")
    with pytest.raises(ModelOutputTruncatedError):
        read_segment({"text": original}, invoke=always_fail)
    assert len(calls) == 2  # 子段再次截断立即结束，不无限细分。


def test_additive_retry_marker_migration_preserves_existing_tasks(tmp_path):
    from agentic_rag.research.store import SCHEMA
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA.replace(", retryable INTEGER NOT NULL DEFAULT 1", ""))
        db.execute("INSERT INTO tasks(key,kind,payload,status,result,updated) VALUES('old','reader','{}','complete','{}',0)")
    store = ResearchStore(path)
    assert store.task("old")["status"] == "complete"
    assert store.task("old")["retryable"] == 1
