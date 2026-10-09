"""会话记忆的确定性回归；不依赖模型联网或真实财经结论。"""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from agentic_rag import conversation as memory
from agentic_rag import cli
from agentic_rag.graph.state import GraphState


@pytest.fixture
def database(tmp_path):
    return memory.ConversationStore(tmp_path / "conversations.sqlite")


@pytest.fixture(autouse=True)
def roomy_settings(monkeypatch):
    monkeypatch.setattr(memory, "settings", replace(memory.settings, llm_context_window=16384))


def record(db, cid, question="分析比亚迪近期新闻", answer="1. 销量增长\n2. 原材料成本下降", run=None, status="complete"):
    revision, _ = db.snapshot(cid)
    return db.append(cid, expected_revision=revision, run_id=run or f"{cid}-{revision}", question=question,
                     resolved_question=question, answer=answer, status=status, resolution={})


def followup(question="分析比亚迪原材料成本下降的影响", quote="原材料成本下降", turn=1, field="answer"):
    return {"mode": "followup", "question": question, "references": [{"turn": turn, "field": field, "quote": quote}]}


def test_initial_question_does_not_need_model_and_persists(database):
    model = Mock(side_effect=AssertionError("首轮不调用模型"))
    session = memory.Conversation(database, "demo", resolver=model)
    prepared = session.prepare("分析比亚迪近期新闻")
    assert prepared["question"] == prepared["user_question"]
    assert prepared["conversation_resolution"]["mode"] == "independent"
    session.remember(prepared, {"generation": "原答案", "generation_complete": True, "generation_grounded": True}, "r1")
    reopened = memory.ConversationStore(database.path)
    assert reopened.turn("demo", 1)["answer"] == "原答案"
    model.assert_not_called()


def test_pronoun_without_history_clarifies(database):
    session = memory.Conversation(database)
    assert session.prepare("它的风险呢")["conversation_resolution"]["mode"] == "clarify"


def test_pronoun_and_ordinal_use_verified_original(database):
    session = memory.Conversation(database, resolver=lambda _: followup())
    record(database, session.id)
    result = session.prepare("刚才第二条具体有什么影响？")
    assert result["question"] == "分析比亚迪原材料成本下降的影响"
    assert result["user_question"] == "刚才第二条具体有什么影响？"
    assert result["conversation_revision"] == 1


@pytest.mark.parametrize("decision", [
    followup(quote="不存在的事实"), followup(turn=999),
    followup(quote="销量增长"),  # 第二条不能绑定第一条。
    {"mode": "independent"},
])
def test_bad_ordinal_or_fabricated_reference_clarifies(database, decision):
    session = memory.Conversation(database, resolver=lambda _: decision)
    record(database, session.id)
    assert session.prepare("解释第二条")["conversation_resolution"]["mode"] == "clarify"


def test_ambiguous_multiple_numbered_sections_clarifies(database):
    session = memory.Conversation(database, resolver=lambda _: followup())
    record(database, session.id, answer="1. 销量\n2. 原材料成本下降\n其他\n1. 新闻\n2. 另一个事件")
    assert session.prepare("第二条呢")["conversation_resolution"]["mode"] == "clarify"


def test_unknown_long_ordinal_is_not_guessed(database):
    session = memory.Conversation(database, resolver=lambda _: followup())
    record(database, session.id)
    assert session.prepare("第十一条呢")["conversation_resolution"]["mode"] == "clarify"


def test_independent_question_cannot_be_rewritten_by_model(database):
    session = memory.Conversation(database, resolver=lambda _: {"mode": "independent", "question": "继续研究比亚迪"})
    record(database, session.id)
    result = session.prepare("解释黄金价格与利率的关系")
    assert result["question"] == "解释黄金价格与利率的关系"
    assert result["conversation_resolution"]["references"] == []


def test_explicit_topic_reset_bypasses_old_history(database):
    model = Mock(side_effect=AssertionError("明确换题不调用模型"))
    session = memory.Conversation(database, resolver=model)
    record(database, session.id)
    assert session.prepare("换个话题，解释通胀")["question"] == "换个话题，解释通胀"
    fresh = memory.Conversation(database, resolver=model)
    assert fresh.prepare("它的风险")["conversation_resolution"]["mode"] == "clarify"


def test_compression_is_bounded_and_retains_exact_original(database):
    session = memory.Conversation(database)
    answer = "1. 第一项\n2. 第二项\n" + "这是一条较长但完整的原始事实。\n" * 800
    for i in range(7):
        record(database, session.id, question=f"第{i}个问题", answer=answer)
    revision, turns = database.snapshot(session.id)
    packet = memory.build_context(turns, budget_bytes=1800, total_turns=revision)
    assert 0 < packet.bytes <= 1800
    assert packet.compressed > 0
    assert packet.compressed + packet.omitted == 7
    assert database.turn(session.id, 1)["answer"] == answer
    for row in packet.records:
        for excerpt in row["answer_excerpts"]:
            assert excerpt in answer


def test_long_atomic_fact_not_cut_mid_sentence():
    text = "数据" * 500 + "。"
    assert memory.excerpts(text, max_bytes=100) == []
    packet = memory.build_context([], budget_bytes=100)
    assert packet.records == [] and packet.bytes == 2


def test_old_turn_explicit_readback_is_scoped(database):
    captured = []
    def resolver(payload):
        captured.append(payload)
        return followup(quote="旧公司", field="question")
    session = memory.Conversation(database, resolver=resolver)
    record(database, session.id, question="分析旧公司")
    for _ in range(105):
        record(database, session.id, question="分析新公司")
    result = session.prepare("第1轮提到的公司现在如何")
    assert result["conversation_resolution"]["mode"] == "followup"
    assert [row["sequence"] for row in captured[0]["history"]] == [1]
    assert session.prepare("第999轮的公司呢")["conversation_resolution"]["mode"] == "clarify"


def test_compressed_ordinal_verified_against_raw(database):
    session = memory.Conversation(database, resolver=lambda _: followup(), recent_turns=0)
    record(database, session.id, answer="1. 销量增长\n2. 原材料成本下降\n" + "这是一段很长的额外解释。\n" * 100)
    result = session.prepare("刚才第二条呢")
    assert result["conversation_resolution"]["mode"] == "followup"
    assert result["conversation_resolution"]["compressed_turns"] == 1


def test_summary_never_inflates_short_turn(database):
    session = memory.Conversation(database)
    record(database, session.id, question="问题", answer="回答")
    _, turns = database.snapshot(session.id)
    packet = memory.build_context(turns, budget_bytes=2000, recent_turns=0)
    assert packet.compressed == 0
    assert packet.records[0]["answer"] == "回答"


def test_full_numbered_quote_from_real_model_is_accepted(database):
    session = memory.Conversation(database, resolver=lambda _: followup(quote="2. 原材料成本下降"))
    record(database, session.id)
    assert session.prepare("刚才第二条呢")["conversation_resolution"]["mode"] == "followup"


def test_new_explicit_date_cannot_be_dropped(database):
    session = memory.Conversation(database, resolver=lambda _: followup(quote="比亚迪", field="question"))
    record(database, session.id)
    assert session.prepare("它在2025年有什么风险")["conversation_resolution"]["mode"] == "clarify"


def test_first_self_contained_why_is_not_mistaken_for_followup(database):
    session = memory.Conversation(database)
    assert session.prepare("为什么利率会影响黄金")["conversation_resolution"]["mode"] == "independent"


def test_immediate_pronoun_cannot_jump_to_older_subject(database):
    session = memory.Conversation(database, resolver=lambda _: followup(quote="比亚迪", field="question"))
    record(database, session.id)
    record(database, session.id, question="分析宁德时代")
    assert session.prepare("它的风险呢")["conversation_resolution"]["mode"] == "clarify"


def test_failed_resolver_cannot_pass_unresolved_pronoun_as_independent(database):
    session = memory.Conversation(database, resolver=lambda _: {"mode": "independent"})
    record(database, session.id)
    assert session.prepare("它的风险呢")["conversation_resolution"]["mode"] == "clarify"


def test_omitted_turn_cannot_be_referenced(database):
    session = memory.Conversation(database, resolver=lambda _: followup(), budget_bytes=350)
    record(database, session.id, answer="特别长的未分段回答" * 2000)
    result = session.prepare("它怎么样")
    assert result["conversation_resolution"]["mode"] == "clarify"


def test_too_long_question_stops_before_model(database):
    model = Mock(side_effect=AssertionError("预算不足不调用"))
    session = memory.Conversation(database, resolver=model, budget_bytes=100)
    result = session.prepare("问题" * 1000)
    assert result["conversation_resolution"]["mode"] == "clarify"
    model.assert_not_called()


def test_model_failure_clarifies_without_losing_history(database):
    model = Mock(side_effect=TimeoutError("no service"))
    session = memory.Conversation(database, resolver=model)
    record(database, session.id)
    result = session.prepare("它的风险呢")
    assert result["conversation_resolution"]["error_type"] == "TimeoutError"
    assert database.snapshot(session.id)[0] == 1


def test_compare_and_swap_and_idempotency(database):
    cid = database.ensure("a")
    kwargs = dict(expected_revision=0, run_id="r", question="问题", resolved_question="问题", answer="答案", status="complete", resolution={})
    assert database.append(cid, **kwargs) == 1
    assert database.append(cid, **kwargs) == 1
    with pytest.raises(memory.ConversationConflict):
        database.append(cid, **{**kwargs, "run_id": "other"})
    other = database.ensure("b")
    with pytest.raises(memory.ConversationConflict):
        database.append(other, **kwargs)
    assert database.snapshot(cid)[0] == 1


@pytest.mark.parametrize("value", ["../private", "", "a' OR 1=1", "a" * 81])
def test_invalid_session_id_rejected(database, value):
    with pytest.raises(ValueError):
        database.ensure(value)


def test_clarification_never_starts_research(database, monkeypatch, capsys):
    session = memory.Conversation(database)
    execute = Mock(side_effect=AssertionError("不应执行图"))
    monkeypatch.setattr(cli, "ask", execute)
    assert not cli.converse(None, "它的风险", session)
    assert "需要澄清" in capsys.readouterr().out
    assert database.turn(session.id, 1)["status"] == "clarification"
    execute.assert_not_called()


def test_each_turn_new_run_and_only_resolved_question_enters_graph(database, monkeypatch):
    session = memory.Conversation(database, resolver=lambda _: followup())
    invocations = []
    def ask(graph, question, **kwargs):
        invocations.append((question, kwargs))
        kwargs["on_result"]({"generation": "1. 销量增长\n2. 原材料成本下降", "generation_complete": True, "generation_grounded": True})
        return True
    monkeypatch.setattr(cli, "ask", ask)
    assert cli.converse(None, "分析比亚迪近期新闻", session)
    assert cli.converse(None, "第二条呢", session)
    assert invocations[0][1]["run_id"] != invocations[1][1]["run_id"]
    assert invocations[1][1]["conversation_input"]["user_question"] == "第二条呢"
    assert "documents" not in invocations[1][1]["conversation_input"]
    assert invocations[1][0] == "分析比亚迪原材料成本下降的影响"


def test_actual_graph_state_contains_conversation_audit_but_no_old_evidence():
    flow = StateGraph(GraphState)
    flow.add_node("answer", lambda state: {"generation": state["question"], "generation_complete": True, "generation_grounded": True})
    flow.add_edge(START, "answer")
    flow.add_edge("answer", END)
    graph = flow.compile(checkpointer=InMemorySaver())
    prepared = {"conversation_id": "c", "conversation_revision": 1, "user_question": "它呢", "conversation_resolution": {"mode": "followup"}}
    result = cli.run_with_trace(graph, "比亚迪的风险", run_id="new-run", conversation_input=prepared)
    saved = graph.get_state({"configurable": {"thread_id": "new-run"}}).values
    assert result["user_question"] == saved["user_question"] == "它呢"
    assert saved["question"] == "比亚迪的风险" and "documents" not in saved


def test_failed_research_does_not_append_answer(database, monkeypatch):
    session = memory.Conversation(database)
    monkeypatch.setattr(cli, "ask", lambda *args, **kwargs: False)
    assert not cli.converse(None, "分析比亚迪", session)
    assert database.snapshot(session.id)[0] == 0


def test_failed_completion_is_labelled_not_certified(database):
    session = memory.Conversation(database)
    prepared = session.prepare("分析公司")
    session.remember(prepared, {"generation": "证据不足", "generation_complete": False, "generation_grounded": True}, "r")
    assert database.turn(session.id, 1)["status"] == "needs_attention"


def test_clarification_reply_can_use_unresolved_question(database):
    session = memory.Conversation(database, resolver=lambda _: {"mode": "clarify", "clarification": "指哪家公司？"})
    record(database, session.id, question="比较比亚迪与宁德时代")
    pending = session.prepare("它的风险呢")
    session.remember(pending, {"generation": "指哪家公司？"}, "clarify")
    session.resolver = lambda _: followup(question="比亚迪的风险是什么", quote="它的风险呢", field="question", turn=2)
    resolved = session.prepare("比亚迪")
    assert resolved["question"] == "比亚迪的风险是什么"


def test_interrupt_in_resolver_never_starts_research(database, monkeypatch, capsys):
    session = memory.Conversation(database, resolver=Mock(side_effect=KeyboardInterrupt()))
    record(database, session.id)
    assert not cli.converse(None, "它呢", session)
    assert "会话理解已中断" in capsys.readouterr().out
    assert database.snapshot(session.id)[0] == 1


def test_sqlite_failure_is_not_silently_treated_as_new_conversation(database, monkeypatch, capsys):
    session = memory.Conversation(database)
    monkeypatch.setattr(database, "snapshot", Mock(side_effect=OSError("read failed")))
    assert not cli.converse(None, "独立问题", session)
    assert "会话读取失败" in capsys.readouterr().out


def test_cli_interactive_memory_reset_and_readback(tmp_path, monkeypatch, capsys):
    from agentic_rag.research.store import ResearchStore
    monkeypatch.setattr(cli, "settings", replace(cli.settings, ollama_warmup_enabled=False,
        knowledge_watch_enabled=False, checkpoint_db=str(tmp_path / "checkpoints.sqlite"),
        conversation_db=str(tmp_path / "conversations.sqlite")))
    monkeypatch.setattr("agentic_rag.research.service.store", lambda: ResearchStore(tmp_path / "research.sqlite"))
    def build(*, checkpointer):
        flow = StateGraph(GraphState)
        flow.add_node("answer", lambda state: {"generation": "1. 销量增长\n2. 原材料成本下降", "generation_complete": True, "generation_grounded": True})
        flow.add_edge(START, "answer")
        flow.add_edge("answer", END)
        return flow.compile(checkpointer=checkpointer)
    monkeypatch.setattr(cli, "build_graph", build)
    monkeypatch.setattr(cli, "unload_model", lambda **kwargs: True)
    monkeypatch.setattr(memory, "resolve_with_model", lambda _: followup())
    questions = iter(["分析比亚迪", "第二条呢", "/history", "/turn 1", "/new", "它呢", "exit"])
    monkeypatch.setattr(cli, "input_with_timeout", lambda *args: next(questions))
    monkeypatch.setattr("sys.argv", ["agentic-rag"])
    cli.main()
    output = capsys.readouterr().out
    assert "追问还原" in output and "已切换新会话" in output and "需要澄清" in output
    store = memory.ConversationStore(tmp_path / "conversations.sqlite")
    with store.connect() as conn:
        rows = conn.execute("SELECT id, revision FROM conversations ORDER BY revision DESC").fetchall()
    assert sorted(row[1] for row in rows) == [1, 2]
    assert store.turn(rows[0][0], 2)["question"] == "第二条呢"


def test_store_failure_does_not_repeat_completed_graph(monkeypatch, capsys):
    result = {"generation": "结果", "generation_complete": True, "generation_grounded": True}
    run = Mock(return_value=result)
    monkeypatch.setattr(cli, "run_with_trace", run)
    assert not cli.ask(None, "问题", on_result=Mock(side_effect=OSError("disk full")))
    run.assert_called_once()
    assert "会话未保存" in capsys.readouterr().out


def test_resolver_does_not_expand_prompt_or_retry_on_transport_failure(monkeypatch):
    import httpx
    from agentic_rag.graph import chains
    runtime = Mock()
    runtime.invoke.side_effect = httpx.ConnectError("offline")
    monkeypatch.setattr(chains, "get_llm", lambda: Mock())
    monkeypatch.setattr(chains, "_structured_runtime", lambda *args: runtime)
    with pytest.raises(chains.ModelAdaptiveRetryError):
        memory.resolve_with_model({"question": "它呢", "history": []})
    runtime.invoke.assert_called_once()
