"""dd4f2e 回归：真实规划语义、长解析错误及每次尝试的实际计量参数。"""

from dataclasses import replace
from uuid import uuid4

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from agentic_rag.graph import chains
from agentic_rag.news_plan import prepare_news_plan, query_covers_term
from agentic_rag.token_usage import UsageCallback, UsageLedger


def source_plan(**changes):
    return {
        "use_knowledge": False, "use_news": True, "use_web": True,
        "task_type": "analysis", "evidence_needs": ["公开政策立场", "双方互动记录"],
        "knowledge_query": "", "news_query": "特朗普 泽连斯基",
        "additional_news_queries": ["特朗普 乌克兰政策", "泽连斯基 特朗普 关系"],
        "news_people": ["特朗普", "泽连斯基"],
        "news_topics": ["美乌关系", "乌克兰战争", "美国大选政策"],
        "news_section": "",
        "news_time": {"mode": "unrestricted", "start": "", "end": "", "reason": ""},
        "web_query": "Trump Zelensky relationship analysis",
        "news_coverage": "broad", "news_result_limit": 10,
        **changes,
    }


def test_observed_source_plan_is_usable_without_identical_topic_labels():
    model = chains.SourcePlan(**source_plan())
    plan = prepare_news_plan(model.news_query, model.news_section, model.news_time.model_dump(),
                             additional_queries=model.additional_news_queries,
                             people=model.news_people, topics=model.news_topics)
    assert len(plan["queries"]) == 4
    assert plan["queries"][:3] == [model.news_query, *model.additional_news_queries]
    assert plan["added_topic_queries"] == ["特朗普 美乌关系"]
    assert plan["deferred_topics"] == ["乌克兰战争", "美国大选政策"]
    assert not plan["error"] and not plan["start"] and not plan["end"]


def test_topic_compiler_honors_and_semantics_order_and_capacity():
    assert query_covers_term("泽连斯基 特朗普 关系", "特朗普 泽连斯基")
    plan = prepare_news_plan("特朗普", "", {}, additional_queries=["泽连斯基"],
                             topics=["特朗普 泽连斯基"])
    assert plan["queries"] == ["特朗普", "泽连斯基", "特朗普 泽连斯基"]
    covered = prepare_news_plan("泽连斯基 特朗普", "", {}, topics=["特朗普 泽连斯基"])
    assert not covered["added_topic_queries"] and not covered["deferred_topics"]
    assert prepare_news_plan("", "", {}, topics=["人工智能"])["queries"] == ["人工智能"]


def test_every_named_entity_must_be_in_an_executed_query():
    with pytest.raises(ValidationError, match="人物=泽连斯基"):
        chains.SourcePlan(**source_plan(news_query="特朗普", additional_news_queries=[]))


def test_query_limits_match_real_server_before_planning_or_network_calls():
    from agentic_rag.news_api import NewsClient
    from agentic_rag.tools.news import NewsSearchInput
    query = " ".join(["term"] * 9)
    with pytest.raises(ValidationError, match="最多8"):
        chains.SourcePlan(**source_plan(news_query=query, additional_news_queries=[], news_people=[]))
    with pytest.raises(ValidationError, match="最多8"):
        NewsSearchInput(semantic_query="问题", queries=[query])
    client = NewsClient(api_key="fake", base_url="https://example.com")
    with pytest.raises(ValueError, match="最多8"):
        client.search(q=query)
    with pytest.raises(ValueError, match="80"):
        client.search(q="test", section="x" * 81)
    with pytest.raises(ValidationError):
        NewsSearchInput(semantic_query="问题", queries=["test"], section="x" * 81)
    plan = prepare_news_plan("特朗普", "", {}, topics=[query])
    assert plan["deferred_topics"] == [query]
    assert plan["queries"] == ["特朗普"]


def test_error_feedback_extracts_field_failure_before_truncating(monkeypatch):
    class Expected(chains.StructuredOutput):
        count: int

    try:
        Expected.model_validate({"count": "wrong"})
    except ValidationError as cause:
        parser_error = OutputParserException("Failed to parse " + "x" * 6000)
        parser_error.__cause__ = cause

    calls = []

    def answer(messages):
        calls.append(messages)
        if len(calls) == 1:
            return {"raw": AIMessage(content='{"count":"wrong"}'), "parsed": None,
                    "parsing_error": parser_error}
        return {"raw": AIMessage(content='{"count":2}'), "parsed": Expected(count=2),
                "parsing_error": None}

    class Fake:
        def with_structured_output(self, schema, include_raw=False):
            return RunnableLambda(answer)

    monkeypatch.setattr(chains, "get_llm", Fake)
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_max_attempts=2))
    assert chains.checked_structured(Expected).invoke("问题").count == 2
    feedback = calls[1][-1].content
    assert "count" in feedback and "int_parsing" in feedback
    assert "x" * 100 not in feedback and len(feedback) < 500


def test_usage_records_actual_attempt_switch_and_model_name():
    ledger = UsageLedger()
    callback = UsageCallback(ledger, operation="规划", thinking_requested=True)
    callback.on_chat_model_start({}, [[HumanMessage(content="问题")]], run_id=uuid4(),
                                 invocation_params={},
                                 metadata={"ls_model_name": "gemma4:26b", "usage_thinking_requested": False})
    call = next(iter(ledger.calls.values()))
    assert call["thinking_requested"] is False and call["model"] == "gemma4:26b"


def test_retry_updates_callback_metadata_without_mutating_original_config(monkeypatch):
    received = []

    def answer(messages, config):
        received.append(config["metadata"]["usage_thinking_requested"])
        if len(received) == 1:
            raise TimeoutError("slow")
        return AIMessage(content="完成")

    original = {"metadata": {"usage_thinking_requested": True, "custom": "keep"}}
    monkeypatch.setattr(chains, "get_llm", lambda: RunnableLambda(answer))
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_reasoning=True,
                                                   llm_max_attempts=2))
    assert chains.checked_text("ResearchAnswer").invoke("问题", config=original) == "完成"
    assert received == [True, False]
    assert original["metadata"] == {"usage_thinking_requested": True, "custom": "keep"}


def test_exact_time_is_not_lost_when_model_supplies_empty_exact_slot():
    time = chains.NewsTimeSuggestion(mode="explicit", start="2026-09-30T20:00:00Z", end="",
                                     published_after="", reason="用户指定")
    assert time.published_after == "2026-09-30T20:00:00Z"
    assert time.start == "2026-10-01"
    with pytest.raises(ValidationError, match="冲突"):
        chains.NewsTimeSuggestion(mode="explicit", start="2026-09-30T20:00:00Z", end="",
                                  published_after="2026-09-30T10:00:00Z", reason="用户指定")


@pytest.mark.parametrize("status", ["running", "completed", "needs_attention"])
def test_recovered_run_does_not_report_previous_failure_as_current(tmp_path, status):
    from types import SimpleNamespace
    from agentic_rag.research.inspection import inspect_research
    from agentic_rag.research.store import ResearchStore
    database = ResearchStore(tmp_path / "research.sqlite")
    database.start_run("run", "问题")
    database.event("run", {"kind": "run_failure", "error_type": "ModelOutputValidationError", "message": "旧错误"})
    database.event("run", {"kind": "token_usage", "event": "session_started", "session_id": "new"})
    database.run_status("run", status)
    report = inspect_research(database, "run", snapshot=SimpleNamespace(values={}, next=(), tasks=()))
    assert report["failure"] is None and report["last_failure"]["message"] == "旧错误"


def test_cli_preserves_actionable_validation_reason():
    from agentic_rag.cli import _run_error_message
    reason = "schema: " + "字段x；" * 40 + "缺少必需的新闻人物=泽连斯基"
    assert _run_error_message(chains.ModelOutputValidationError(reason)) == reason


def test_news_service_503_is_not_misreported_as_ollama_failure():
    from agentic_rag.cli import _run_error_message
    from agentic_rag.news_api import NewsAPIError
    error = NewsAPIError("新闻数据库暂时不可用", status_code=503, error_code="database_unavailable")
    assert _run_error_message(error) == str(error)
    wrapper = RuntimeError("外层异常")
    wrapper.__cause__ = error
    assert _run_error_message(wrapper) == str(error)


def test_completed_resume_works_offline_without_watcher_or_state_writes(monkeypatch, capsys):
    from types import SimpleNamespace
    from agentic_rag import cli
    from agentic_rag.research import inspection, service
    snapshot = SimpleNamespace(values={"question": "问题", "generation": "已保存答案"}, next=(), tasks=())
    monkeypatch.setattr(inspection, "read_snapshot", lambda _: snapshot)
    monkeypatch.setattr(service, "store", lambda: SimpleNamespace(token_events=lambda _: []))
    monkeypatch.setattr("sys.argv", ["agentic-rag", "--resume", "saved"])
    monkeypatch.setattr(cli, "settings", replace(cli.settings, knowledge_watch_enabled=True, ollama_warmup_enabled=True))
    def unexpected(*args, **kwargs):
        pytest.fail("展示保存结果不应连接模型、启动监听或进入研究执行")
    for method in ("warm_up_model", "unload_model", "start_knowledge_watcher", "ask"):
        monkeypatch.setattr(cli, method, unexpected)
    cli.main()
    output = capsys.readouterr().out
    assert "已保存答案" in output and "不连接模型" in output


@pytest.mark.parametrize("retry_failed", [False, True])
def test_resume_with_pending_tasks_or_explicit_retry_still_checks_model(monkeypatch, retry_failed):
    from types import SimpleNamespace
    from agentic_rag import cli
    from agentic_rag.research import inspection, service
    snapshot = SimpleNamespace(values={"generation": "草稿"}, next=(), tasks=(SimpleNamespace(result={}),))
    monkeypatch.setattr(inspection, "read_snapshot", lambda _: snapshot)
    monkeypatch.setattr(service, "store", lambda: object())
    monkeypatch.setattr("sys.argv", ["agentic-rag", "--resume", "run", *(["--retry-failed"] if retry_failed else [])])
    monkeypatch.setattr(cli, "settings", replace(cli.settings, knowledge_watch_enabled=False, ollama_warmup_enabled=True))
    calls = []
    monkeypatch.setattr(cli, "warm_up_model", lambda **_: calls.append("warmup") or False)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1 and calls == ["warmup"]
