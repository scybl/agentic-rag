"""真实回调链、并发归属、失败/重试、恢复和终端统计；不依赖在线模型。"""

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from ollama import ResponseError

from agentic_rag import cli
from agentic_rag.graph import chains
from agentic_rag.research.scheduler import Scheduler, TaskSpec
from agentic_rag.research.store import ResearchStore
from agentic_rag.token_usage import (
    UsageLedger, active_usage, measured_usage, meter_node, print_usage_summary, usage_session,
)


class CountedModel(BaseChatModel):
    """让 LangChain 真实触发回调，不用手工模拟结束回调代替接线测试。"""

    @property
    def _llm_type(self):
        return "counted-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content='{"ok": true}', usage_metadata={
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
            "output_token_details": {"reasoning": 1},
        })
        return ChatResult(generations=[ChatGeneration(message=message, generation_info={
            "prompt_eval_count": 10, "eval_count": 4, "prompt_eval_cached_count": 3,
        })])


def model_chain(parser=None):
    return chains._with_model_retry(
        ChatPromptTemplate.from_messages([("system", "规则"), ("human", "{question}")])
        | CountedModel() | (parser or StrOutputParser()), operation="测试调用")


def test_counts_are_measured_and_subsets_are_not_added_twice():
    result = measured_usage({"prompt_eval_count": 100, "eval_count": 20, "prompt_eval_cached_count": 40},
                            {"input_tokens": 999, "output_tokens": 888,
                             "output_token_details": {"reasoning": 7}})
    assert result == {"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 7,
                      "visible_output_tokens": 13, "cached_input_tokens": 40}
    assert measured_usage() == dict.fromkeys(result)
    assert measured_usage({"prompt_eval_count": True, "eval_count": -2}) == dict.fromkeys(result)
    invalid = measured_usage({"prompt_eval_count": 2, "eval_count": 3, "prompt_eval_cached_count": 5},
                             {"output_token_details": {"reasoning": 4}})
    assert invalid["reasoning_tokens"] is None and invalid["cached_input_tokens"] is None
    assert measured_usage(usage={"input_tokens": 2, "output_tokens": 3})["output_tokens"] == 3


@pytest.mark.parametrize("parser", [StrOutputParser(), JsonOutputParser()])
def test_usage_survives_output_parser_and_does_not_persist_content(parser):
    events, ledger = [], UsageLedger()
    ledger.persist = events.append
    chain = model_chain(parser)
    with usage_session(ledger):
        meter_node("route", lambda state: chain.invoke(state))({"question": "私人问题"})
    report = ledger.report()
    assert report["current"]["total_tokens"] == 14
    assert report["current"]["reasoning_tokens"] == 1
    assert report["current"]["visible_output_tokens"] == 3
    assert report["steps"][0]["calls"] == 1
    call = next(iter(ledger.calls.values()))
    assert call["input_characters"] == {"system": 2, "human": 4}
    assert "私人问题" not in json.dumps(events, ensure_ascii=False)
    assert '{"ok": true}' not in json.dumps(events)
    assert active_usage.get() is None


def test_parser_failure_still_counts_model_generation():
    ledger = UsageLedger()
    def fail(value):
        raise ValueError("schema error")
    with usage_session(ledger), pytest.raises(ValueError):
        meter_node("route", lambda state: model_chain(RunnableLambda(fail)).invoke(state))({"question": "问"})
    assert ledger.report()["current"]["total_tokens"] == 14
    assert ledger.report()["steps"][0]["status"] == "failed"


def test_empty_output_retry_counts_every_completed_response(monkeypatch):
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_max_attempts=2))
    attempts = []
    def parse(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise chains.EmptyModelOutputError("empty")
        return value
    ledger = UsageLedger()
    with usage_session(ledger):
        model_chain(RunnableLambda(parse)).invoke({"question": "问"})
    assert len(attempts) == 2
    assert ledger.report()["current"]["calls"] == 2
    assert ledger.report()["current"]["total_tokens"] == 28


def test_failed_transport_is_unknown_not_zero(monkeypatch):
    class BrokenModel(CountedModel):
        def _generate(self, *args, **kwargs):
            raise ResponseError("busy", 503)
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_max_attempts=1))
    ledger = UsageLedger()
    with usage_session(ledger), pytest.raises(ResponseError):
        chains._with_model_retry(BrokenModel()).invoke("问")
    stats = ledger.report()["current"]
    assert stats["calls"] == stats["unknown"] == stats["failed"] == 1
    assert not stats["complete"]
    output = io.StringIO()
    with redirect_stdout(output):
        print_usage_summary(ledger)
    assert "输入：未报告" in output.getvalue()
    assert "不是完整总量" in output.getvalue()


def test_scheduler_propagates_usage_and_task_context_and_reuse_costs_zero(tmp_path):
    db = ResearchStore(tmp_path / "research.db")
    ledger = UsageLedger(persist=lambda e: db.event("r", e))
    specs = [TaskSpec(str(i), "reader", {"question": "问"}) for i in range(4)]
    chain = model_chain(JsonOutputParser())
    def work(state):
        return Scheduler(db, "r", workers=3).run(specs, {"reader": chain.invoke})
    with usage_session(ledger):
        meter_node("read_documents", work)({})
        meter_node("read_documents", work)({})
    report = ledger.report()
    assert [s["calls"] for s in report["steps"]] == [4, 0]
    assert report["current"]["total_tokens"] == 56
    assert {c["task_id"] for c in ledger.calls.values()} == {"0", "1", "2", "3"}
    assert all(c["task_attempt"] == 1 and c["step"] == "read_documents" for c in ledger.calls.values())
    other = UsageLedger()
    with usage_session(other):
        chain.invoke({"question": "第二问"})
    assert other.report()["current"]["calls"] == 1
    assert ledger.report()["current"]["calls"] == 4


def test_task_retry_and_specialist_second_pass_each_count(tmp_path):
    db = ResearchStore(tmp_path / "research.db")
    ledger = UsageLedger()
    attempts = []
    chain = model_chain(JsonOutputParser())
    def work(payload):
        attempts.append(1)
        result = chain.invoke({"question": "专题分析"})
        if len(attempts) == 1:
            raise ValueError("阅读结果校验不通过")
        # 同一专题回读原文之后的第二次推理，也必须累计。
        return chain.invoke({"question": "根据回读结果复核"}) or result
    with usage_session(ledger):
        meter_node("dispatch_specialists", lambda state: Scheduler(db, "r", attempts=2).run(
            [TaskSpec("topic", "specialist", {})], {"specialist": work}))({})
    assert ledger.report()["current"]["total_tokens"] == 42
    assert [c["task_attempt"] for c in ledger.calls.values()] == [1, 2, 2]


def test_usage_history_is_not_truncated_to_last_100_events(tmp_path):
    db = ResearchStore(tmp_path / "research.db")
    ledger = UsageLedger(persist=lambda e: db.event("r", e))
    for i in range(120):
        call = ledger.begin_call()
        ledger.finish_call(call, metadata={"prompt_eval_count": 10, "eval_count": 4})
    history = db.token_events("r")
    assert len(history) == 240 and len(db.events("r")) == 100
    restored = UsageLedger(history=history + history)
    assert restored.report()["current"]["calls"] == 0
    assert restored.report()["cumulative"]["calls"] == 120
    assert restored.report()["cumulative"]["total_tokens"] == 1680


def test_incomplete_attempt_and_late_result_are_not_lost(tmp_path):
    db = ResearchStore(tmp_path / "research.db")
    ledger = UsageLedger(persist=lambda e: db.event("r", e))
    call = ledger.begin_call()
    restored = UsageLedger(history=db.token_events("r"))
    assert restored.report()["cumulative"]["pending"] == 1
    assert not restored.report()["cumulative"]["complete"]
    ledger.finish_call(call, metadata={"prompt_eval_count": 10, "eval_count": 4})
    ledger.finish_call(call, metadata={"prompt_eval_count": 1000, "eval_count": 1000})
    assert UsageLedger(history=db.token_events("r")).report()["cumulative"]["total_tokens"] == 14


def test_old_unrecorded_usage_gap_survives_a_second_resume():
    events = []
    with usage_session(UsageLedger(historical_gap=True, persist=events.append)):
        pass
    assert UsageLedger(history=events).report()["historical_gap"]


def test_ledger_persistence_failure_never_retries_model():
    def broken(event):
        raise OSError("disk full")
    ledger = UsageLedger(persist=broken)
    with usage_session(ledger):
        model_chain().invoke({"question": "问"})
    assert ledger.report()["current"]["calls"] == 1
    assert ledger.report()["persistence_errors"] == 4  # 执行开始/结束及请求开始/结束。


def test_graph_stream_wakes_before_node_finishes_and_summary_is_after_answer():
    chain = model_chain()
    graph = StateGraph(dict)
    def route(state):
        chain.invoke({"question": "问"})
        return {"generation": "最终答案内容"}
    graph.add_node("route", meter_node("route", route))
    graph.add_edge(START, "route")
    graph.add_edge("route", END)
    ledger = UsageLedger()
    with usage_session(ledger):
        events = list(graph.compile().stream({"question": "问"}, stream_mode=["updates", "custom"]))
    assert next(mode for mode, event in events) == "custom"
    assert ledger.report()["steps"][0]["calls"] == 1
    output = io.StringIO()
    with redirect_stdout(output):
        assert cli.ask(graph.compile(), "问", verbose=True)
    text = output.getvalue()
    assert text.index("[Token ·") < text.index("最终回答") < text.index("本次问题 Token 统计")
    assert "总计 14 token" in text


def test_cli_failure_still_prints_tokens():
    class Graph:
        def stream(self, state, **kwargs):
            model_chain().invoke(state)
            raise ValueError("after model")
            yield
    output = io.StringIO()
    with redirect_stdout(output):
        assert not cli.ask(Graph(), "问")
    assert "Token 最终合计" in output.getvalue()
    assert "总计 14 token" in output.getvalue()


def test_cli_interrupt_still_prints_tokens():
    class Graph:
        def stream(self, state, **kwargs):
            model_chain().invoke(state)
            raise KeyboardInterrupt()
            yield
    output = io.StringIO()
    with redirect_stdout(output):
        assert not cli.ask(Graph(), "问")
    assert "总计 14 token" in output.getvalue()


def test_warmup_tokens_are_separate_and_unknown_is_labelled():
    class Client:
        def generate(self, **kwargs):
            return {"prompt_eval_count": 2, "eval_count": 1}
    ledger = UsageLedger()
    output = io.StringIO()
    with usage_session(ledger), redirect_stdout(output):
        assert cli.warm_up_model(client=Client(), verbose=True)
    assert ledger.report()["current"]["calls"] == 0
    assert "模型预热 Token（不计入单次问题）" in output.getvalue()
    assert "总计 3 token" in output.getvalue()
    assert "思考 / 可见输出：服务未单列" in output.getvalue()


def test_completed_resume_shows_historical_usage_without_new_charge(tmp_path, monkeypatch):
    from agentic_rag.research import service
    db = ResearchStore(tmp_path / "research.db")
    monkeypatch.setattr(service, "store", lambda: db)
    ledger = UsageLedger(persist=lambda e: db.event("r", e))
    call = ledger.begin_call()
    ledger.finish_call(call, metadata={"prompt_eval_count": 10, "eval_count": 4})
    class CompletedGraph:
        def get_state(self, config):
            return SimpleNamespace(values={"question": "问", "generation": "保存结果",
                "generation_complete": True, "generation_grounded": True}, next=())
        def stream(self, *args, **kwargs):
            raise AssertionError("已完成研究不能重新推理")
    output = io.StringIO()
    with redirect_stdout(output):
        assert cli.ask(CompletedGraph(), "", run_id="r", resume=True, verbose=True)
    text = output.getvalue()
    assert "本次执行新增：输入 0 + 生成 0" in text
    assert "该研究历史累计（含本次）：输入 10 + 生成 4" in text
