"""紧凑数字视图，以及路由结果写入后中断的真实SQLite恢复回归。"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from agentic_rag import cli
from agentic_rag.research.inspection import has_pending_work, inspect_research, print_inspection
from agentic_rag.research.store import ResearchStore
from agentic_rag.token_usage import UsageLedger, usage_session, print_usage_summary


def test_default_trace_keeps_actual_parameters_and_errors_not_long_prose(capsys):
    trace = cli.TracePrinter()
    long_reason = "这是一大段对数据源选择的解释。" * 30
    trace.show("route", {"selected_sources": ["news_api"], "plan_summary": long_reason,
        "source_queries": {"news_api": "美联储 利率"}, "evidence_needs": ["通胀", "就业"],
        "news_search_plan": {"suggested_time": {"start": "2026-03-30", "end": "2026-09-30"},
                             "start": "2026-03-30", "end": "2026-09-30"}}, {})
    trace.show_event({"kind": "news_error", "message": "HTTP 503", "partial_count": 20})
    text = capsys.readouterr().out
    assert long_reason not in text and "因素 2" in text and "美联储 利率" in text
    assert "2026-03-30~2026-09-30" in text and "HTTP 503" in text and "已保留 20" in text
    assert len(text.splitlines()) <= 5


def test_compact_summary_has_one_total_and_keeps_unknown_and_failure(capsys):
    ledger = UsageLedger()
    with usage_session(ledger):
        call = ledger.begin_call()
        ledger.finish_call(call, metadata={"prompt_eval_count": 10, "eval_count": 4}, elapsed=2)
        call = ledger.begin_call()
        ledger.finish_call(call, failed=True, elapsed=1)
    print_usage_summary(ledger, compact=True)
    text = capsys.readouterr().out
    assert text.count("Token 最终合计") == 1 and "已知小计（不是完整总量） 14 token" in text
    assert "失败 1" in text and "未知用量 1" in text and "[模型异常]" in text
    assert "输入组成" not in text and len(text.splitlines()) <= 6


def test_compact_suggestions_are_not_claimed_as_executed(capsys):
    cli.TracePrinter().show("assess_evidence", {"evidence_assessment": {"ready": True},
        "pending_news_queries": ["非农"], "next_action": "generate"}, {})
    assert "建议，未执行" in capsys.readouterr().out


def test_compact_tool_failure_keeps_error_type_and_degradation(capsys):
    trace = cli.TracePrinter()
    trace.show_event({"kind": "tool", "tool": "search_news", "phase": "started",
                      "caller": "主流程/首次证据收集", "reason": "取得近期事实"})
    trace.show_event({"kind": "tool", "tool": "search_web", "phase": "failed",
                      "error_type": "ToolException", "error": "搜索故障"})
    trace.show_event({"kind": "tool", "tool": "search_news", "phase": "finished",
                      "status": "degraded", "count": 1, "elapsed": 0.2, "warnings": ["只有摘要"]})
    text = capsys.readouterr().out
    assert "主流程/首次证据收集" in text and "取得近期事实" in text
    assert "ToolException" in text and "搜索故障" in text
    assert "降级" in text and "只有摘要" in text
    assert len(text.splitlines()) == 4


def test_compact_trace_shows_dynamic_model_adjustment(capsys):
    trace = cli.TracePrinter()
    trace.show_event({
        "kind": "tool", "tool": "plan_model_retry", "phase": "finished",
        "status": "adjusted", "count": 1, "elapsed": 0.01,
        "action": "increase_output_budget", "retry": True,
        "plan_reason": "输出达到8192上限，临时扩容",
        "adjustments": {"num_predict": 10240, "num_ctx": 16384,
                        "timeout_seconds": 300, "reasoning": True},
        "warnings": [],
    })
    text = capsys.readouterr().out
    assert all(value in text for value in (
        "动态调整", "increase_output_budget", "10240", "16384", "300s", "思考 开",
    ))


@pytest.mark.parametrize('verbose', [False, True])
def test_news_trace_distinguishes_candidates_from_reading_and_shows_weights(capsys, verbose):
    trace = cli.TracePrinter(verbose=verbose)
    trace.show_event({'kind': 'news_ranked', 'candidate_count': 150, 'raw_count': 170,
                      'selected_count': 5, 'deferred_count': 145, 'retrieval_complete': False,
                      'vector_cache_used': False, 'ranking_method': 'tfidf'})
    trace.show_event({'kind': 'news_selected', 'article_id': 'tail', 'title': '最后一页的重要报道',
                      'priority': {'rank': 1, 'score': 0.95}, 'selection_reason': '按阅读优先级并兼顾查询覆盖'})
    output = capsys.readouterr().out
    assert all(word in output for word in ('150', '145', '未取完', 'tfidf', '0.95', 'tail', '按阅读优先级'))
    assert '保持 API 顺序' not in output


def test_all_candidate_weights_are_persisted_and_visible_by_research_id(tmp_path, capsys):
    from agentic_rag.graph.nodes import _source_writer
    from unittest.mock import patch
    database = ResearchStore(tmp_path / 'research.sqlite')
    database.start_run('news-run', '新闻问题')
    with patch('agentic_rag.research.service.store', return_value=database):
        emit = _source_writer({'run_id': 'news-run', 'reading_recipe': 'test'})
        for offset in range(0, 125, 50):
            emit({'kind': 'news_candidates', 'offset': offset,
                  'candidates': [{'article_id': str(i), 'priority': {'rank': i + 1, 'score': 0.1},
                                  'selected_for_reading': i < 5} for i in range(offset, min(125, offset + 50))]})
        emit({'kind': 'news_ranked', 'candidate_count': 125, 'selected_count': 5,
              'deferred_count': 120, 'retrieval_complete': True})
    report = inspect_research(database, 'news-run', snapshot=SimpleNamespace(values={}, next=(), tasks=()))
    candidates = [c for e in report['events'] if e['kind'] == 'news_candidates' for c in e['candidates']]
    assert len(candidates) == 125 and candidates[-1]['priority']['rank'] == 125
    print_inspection(report)
    output = capsys.readouterr().out
    assert '125' in output and '120' in output and '已取完' in output


def test_resume_after_result_write_before_checkpoint_commit_does_not_rerun_router(tmp_path, capsys):
    calls = []
    def route(state):
        calls.append("route")
        return {"question": state["question"], "source_queries": {"news_api": "美联储"}}
    def collect(state):
        calls.append("collect")
        return {"generation": "已恢复后续执行", "generation_complete": True, "generation_grounded": True}
    def build(saver, stop=False):
        flow = StateGraph(dict)
        flow.add_node("route", route)
        flow.add_node("collect", collect)
        flow.add_edge(START, "route")
        flow.add_edge("route", "collect")
        flow.add_edge("collect", END)
        return flow.compile(checkpointer=saver, interrupt_before=["collect"] if stop else None)
    class InterruptCommit(SqliteSaver):
        def put(self, config, checkpoint, metadata, new_versions):
            if metadata.get("step", -1) >= 1:
                raise KeyboardInterrupt("路由结果已保存，但下一步检查点未提交")
            return super().put(config, checkpoint, metadata, new_versions)
    path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "r"}}
    with pytest.raises(KeyboardInterrupt), InterruptCommit.from_conn_string(path) as saver:
        list(build(saver, stop=True).stream({"question": "测试"}, config))
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build(saver)
        snapshot = graph.get_state(config)
        assert snapshot.next == () and snapshot.tasks[0].name == "route"
        assert has_pending_work(snapshot) and not snapshot.values.get("generation")
        database = ResearchStore(tmp_path / "research.sqlite")
        database.start_run("r", "测试")
        database.run_status("r", "interrupted")
        report = inspect_research(database, "r", snapshot=snapshot)
        assert report["staged_nodes"] == ["route"] and not report["has_answer"]
        assert report["counts"]["tool_calls"] == 0
        print_inspection(report)
        assert "已保存节点结果、等待流程推进：route" in capsys.readouterr().out
        result = cli.run_with_trace(graph, "", run_id="r", resume=True)
    assert result["generation"] == "已恢复后续执行" and calls == ["route", "collect"]


def test_checkpoint_without_work_and_without_answer_is_not_completed():
    class Graph:
        def get_state(self, config):
            return SimpleNamespace(values={"question": "未完成"}, next=(), tasks=())
        def stream(self, *args, **kwargs):
            raise AssertionError("没有可推进任务")
    with pytest.raises(ValueError, match="不能判定为已完成"):
        cli.run_with_trace(Graph(), "", run_id="r", resume=True)


def test_failure_diagnostics_do_not_hide_original_error(tmp_path, monkeypatch, capsys):
    database = ResearchStore(tmp_path / "research.sqlite")
    database.start_run("r", "问题")
    def fail(*args, **kwargs):
        raise OSError("diagnostics unavailable")
    monkeypatch.setattr(cli, "inspect_research", fail)
    cli.show_failure_context(database, "r", KeyboardInterrupt())
    assert database.events("r")[-1]["error_type"] == "KeyboardInterrupt"
    assert "原错误：收到中断信号" in capsys.readouterr().out
