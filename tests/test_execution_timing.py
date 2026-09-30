"""单调时钟计时、异常/中断、并发口径、持久化恢复及终端接线。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic_rag import cli, token_usage as usage
from agentic_rag.research.store import ResearchStore


def test_step_and_session_time_include_zero_model_work_and_ignore_wall_clock_changes(monkeypatch):
    ticks = iter([10, 11, 13.5, 16])
    monkeypatch.setattr(usage, "perf_counter", lambda: next(ticks))
    with patch.object(usage.time, "time", side_effect=[1000, 5]):
        ledger = usage.UsageLedger()
        with usage.usage_session(ledger):
            usage.meter_node("collect_sources", lambda state: state)({})
    report = ledger.report()
    assert report["steps"][0]["elapsed"] == 2.5
    assert report["steps"][0]["calls"] == 0
    assert report["timing"]["current"]["elapsed_seconds"] == 6
    assert report["timing"]["current"]["complete"]


@pytest.mark.parametrize("error,status", [(ValueError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_step_and_session_are_timed_even_on_failure_or_interrupt(monkeypatch, error, status):
    ticks = iter([10, 11, 13, 15])
    monkeypatch.setattr(usage, "perf_counter", lambda: next(ticks))
    ledger = usage.UsageLedger()
    def fail(state):
        raise error("test")
    with pytest.raises(error), usage.usage_session(ledger):
        usage.meter_node("generate", fail)({})
    assert ledger.report()["steps"][0]["status"] == status
    assert ledger.sessions[ledger.session_id]["status"] == status
    assert ledger.report()["timing"]["current"]["elapsed_seconds"] == 5


def test_concurrent_request_sum_is_not_execution_wall_time(monkeypatch, capsys):
    ticks = iter([10, 15])
    monkeypatch.setattr(usage, "perf_counter", lambda: next(ticks))
    ledger = usage.UsageLedger()
    with usage.usage_session(ledger):
        a, b = ledger.begin_call(), ledger.begin_call()
        for call in [a, b]:
            ledger.finish_call(call, metadata={"prompt_eval_count": 1, "eval_count": 1}, elapsed=4)
    stats = ledger.report()["timing"]["current"]
    assert stats["elapsed_seconds"] == 5 and stats["model_request_seconds"] == 8
    usage.print_usage_summary(ledger)
    output = capsys.readouterr().out
    assert "总耗时（不含预热与等待输入）：5.00 秒" in output
    assert "本次模型请求耗时之和：8.00 秒" in output
    assert "并发可能重叠，不是总耗时" in output


def test_timing_restore_excludes_pause_and_history_is_not_counted_in_process_twice(tmp_path, monkeypatch):
    database = ResearchStore(tmp_path / "research.sqlite")
    ticks = iter([0, 2, 1000, 1003])
    monkeypatch.setattr(usage, "perf_counter", lambda: next(ticks))
    process = usage.UsageLedger()
    first = usage.UsageLedger(persist=lambda e: database.event("r", e), observer=process.observe)
    with usage.usage_session(first):
        pass
    history = database.token_events("r")
    second = usage.UsageLedger(history=history + history[::-1], observer=process.observe)
    with usage.usage_session(second):
        pass
    assert second.report()["timing"]["current"]["elapsed_seconds"] == 3
    assert second.report()["timing"]["cumulative"]["elapsed_seconds"] == 5
    assert process.report()["timing"]["cumulative"]["elapsed_seconds"] == 5
    assert second.report()["has_history"]


def test_old_or_unfinished_timing_is_unknown_not_zero():
    events = [{"kind": "token_usage", "event": "session_started", "session_id": "old"},
              {"kind": "token_usage", "event": "step_finished", "session_id": "old", "step_id": "s",
               "step": "route", "status": "completed"}]
    report = usage.UsageLedger(history=events).report()
    assert not report["timing"]["cumulative"]["complete"]
    assert report["timing"]["cumulative"]["unknown_sessions"] == 1
    assert report["timing"]["steps"][0]["elapsed_seconds"] is None
    assert not usage.UsageLedger(historical_gap=True).report()["timing"]["cumulative"]["complete"]


def test_finished_steps_do_not_regress_when_events_are_replayed_out_of_order():
    started = {"kind": "token_usage", "event": "step_started", "session_id": "old", "step_id": "s", "step": "route", "status": "running"}
    finished = {**started, "event": "step_finished", "status": "completed", "elapsed": 2}
    report = usage.UsageLedger(history=[finished, started]).report()
    assert report["timing"]["steps"][0]["elapsed_seconds"] == 2
    assert report["timing"]["steps"][0]["status"] == "completed"


@pytest.mark.parametrize("value,expected", [(None, "未记录"), (float('nan'), "未记录"),
    (-1, "未记录"), (True, "未记录"), (0, "0.00 秒"), (443.522, "7 分 23.52 秒"), (3601, "1 小时 0 分 01.00 秒")])
def test_duration_format(value, expected):
    assert usage.format_duration(value) == expected


def test_cli_warmup_is_separate_and_question_summary_is_after_answer(capsys):
    class Client:
        def generate(self, **kwargs):
            return {"prompt_eval_count": 2, "eval_count": 1}
    process = usage.UsageLedger()
    assert cli.warm_up_model(client=Client(), session_ledger=process, verbose=True)
    warmup_output = capsys.readouterr().out
    assert "本次预热耗时" in warmup_output
    assert process.report()["timing"]["cumulative"]["sessions"] == 1
    class Graph:
        def stream(self, *args, **kwargs):
            yield "updates", {"generate": {"generation": "计时测试答案"}}
    assert cli.ask(Graph(), "测试", session_ledger=process, verbose=True)
    text = capsys.readouterr().out
    assert text.index("计时测试答案") < text.index("耗时统计") < text.index("Token 最终合计")
    assert "本程序累计执行耗时（含预热，不含等待输入）" in text
    assert process.report()["timing"]["cumulative"]["sessions"] == 2


def test_completed_resume_has_new_display_time_but_no_recomputed_historical_time(tmp_path, monkeypatch, capsys):
    from agentic_rag.research import service
    database = ResearchStore(tmp_path / "research.sqlite")
    monkeypatch.setattr(service, "store", lambda: database)
    database.event("r", {"kind": "token_usage", "event": "session_started", "session_id": "old"})
    class Graph:
        def get_state(self, config):
            return SimpleNamespace(values={"question": "测试", "generation": "历史回答",
                "generation_complete": True, "generation_grounded": True}, next=())
        def stream(self, *args, **kwargs):
            raise AssertionError("不应重新运行旧研究")
    assert cli.ask(Graph(), "", run_id="r", resume=True, verbose=True)
    text = capsys.readouterr().out
    assert "本研究累计执行耗时" in text and "已知小计" in text
    report = usage.UsageLedger(history=database.token_events("r")).report()
    assert report["timing"]["cumulative"]["unknown_sessions"] == 1


def test_agent_completion_displays_saved_elapsed_without_inventing_time_for_reuse(capsys):
    trace = cli.TracePrinter(verbose=True)
    trace.show_event({"kind": "research", "event": "completed", "task_id": "a", "role": "reader", "elapsed": 65.5})
    assert "任务耗时：1 分 05.50 秒" in capsys.readouterr().out
    trace.show_event({"kind": "research", "event": "reused", "task_id": "b", "role": "reader"})
    assert "任务耗时" not in capsys.readouterr().out
