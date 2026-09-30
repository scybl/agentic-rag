"""记录真实模型用量与执行耗时；缺失不当零，并发时长不冒充墙钟总时长。"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import queue
import threading
import time
import uuid
import math
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler


active_usage = ContextVar("active_token_usage", default=None)
active_step = ContextVar("active_token_step", default=None)
active_task = ContextVar("active_token_task", default=None)
active_notify = ContextVar("active_token_notify", default=None)

STEP_LABELS = {
    "initialize_research": "初始化研究", "route": "规划数据源",
    "recall_memory": "召回长期记忆", "collect_sources": "收集多源证据",
    "read_documents": "分段阅读/复用", "grade_documents": "筛选相关证据",
    "assess_evidence": "检查证据充分性", "dispatch_specialists": "专题分析/回读",
    "supplement_sources": "补充检索", "generate": "生成答案",
    "revise_answer": "重写答案", "evaluate_generation": "核验答案",
    "warmup": "模型预热", "unscoped": "独立模型调用",
}


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def measured_usage(metadata=None, usage=None):
    """Ollama 原始计数优先；只接受明确字段，不通过文本长度估算。"""
    metadata, usage = metadata or {}, usage or {}
    incoming = _count(metadata.get("prompt_eval_count"))
    outgoing = _count(metadata.get("eval_count"))
    if incoming is None:
        incoming = _count(usage.get("input_tokens"))
    if outgoing is None:
        outgoing = _count(usage.get("output_tokens"))
    details = usage.get("output_token_details") or {}
    reasoning = _count(details.get("reasoning"))
    cached = _count(metadata.get("prompt_eval_cached_count"))
    if cached is None:
        cached = _count((usage.get("input_token_details") or {}).get("cache_read"))
    # 分项不允许大于总项；不可信细分标为未知，但保留可信总数。
    if outgoing is None or (reasoning is not None and reasoning > outgoing):
        reasoning = None
    if incoming is None or (cached is not None and cached > incoming):
        cached = None
    return {"input_tokens": incoming, "output_tokens": outgoing,
            "reasoning_tokens": reasoning,
            "visible_output_tokens": outgoing - reasoning if reasoning is not None else None,
            "cached_input_tokens": cached}


def summarize(calls):
    calls = list(calls)
    keys = ("input_tokens", "output_tokens", "reasoning_tokens", "visible_output_tokens", "cached_input_tokens")
    result = {key: sum(c.get(key) or 0 for c in calls) for key in keys}
    result.update(calls=len(calls), failed=sum(c.get("status") == "failed" for c in calls),
                  pending=sum(c.get("status") == "pending" for c in calls),
                  unknown=sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in calls),
                  reasoning_unknown=sum(c.get("reasoning_tokens") is None for c in calls),
                  cached_unknown=sum(c.get("cached_input_tokens") is None for c in calls))
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    result["complete"] = result["unknown"] == 0 and result["pending"] == 0
    # JSON 报表也不把服务没有提供的细分写成 0；主项保留已知小计及 complete 标记。
    if result["reasoning_unknown"]:
        result["reasoning_tokens"] = result["visible_output_tokens"] = None
    if result["cached_unknown"]:
        result["cached_input_tokens"] = None
    return result


def _seconds(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def summarize_timing(sessions, calls):
    """只累加已保存的各次执行时长，不包含恢复之间的停顿，不倒推旧记录。"""
    sessions, calls = list(sessions), list(calls)
    unknown = sum(_seconds(s.get("elapsed")) is None for s in sessions)
    request_unknown = sum(_seconds(c.get("elapsed")) is None for c in calls)
    return {
        "elapsed_seconds": round(sum(_seconds(s.get("elapsed")) or 0 for s in sessions), 3),
        "sessions": len(sessions), "unknown_sessions": unknown, "complete": unknown == 0,
        "model_request_seconds": round(sum(_seconds(c.get("elapsed")) or 0 for c in calls), 3),
        "unknown_model_requests": request_unknown,
    }


def format_duration(seconds):
    seconds = _seconds(seconds)
    if seconds is None:
        return "未记录"
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    minutes, remaining = divmod(round(seconds, 2), 60)
    hours, minutes = divmod(int(minutes), 60)
    return (f"{hours} 小时 " if hours else "") + f"{minutes} 分 {remaining:05.2f} 秒"


class UsageLedger:
    """线程安全的本次与历史账本；原始事件可落盘，按调用 ID 去重重放。"""

    def __init__(self, *, history=(), persist=None, historical_gap=False, observer=None):
        history = list(history)
        self.session_id = uuid.uuid4().hex
        self.persist = persist
        self.observer = observer
        self.historical_gap = historical_gap or any(e.get("historical_gap") for e in history)
        self.calls, self.steps, self.sessions = {}, {}, {}
        self.lock = threading.RLock()
        self.queue = queue.SimpleQueue()
        self.persistence_errors = 0
        for event in history:
            self._apply(event)

    def _apply(self, event):
        if event.get("kind") != "token_usage":
            return
        event_type = event.get("event")
        if event_type in {"call_started", "call_finished"}:
            key = event["call_id"]
            # 重复或乱序重放不能把已完成调用退回 pending。
            if event_type == "call_started" and self.calls.get(key, {}).get("status") in {"completed", "failed"}:
                return
            self.calls[key] = {**self.calls.get(key, {}), **event}
        elif event_type in {"step_started", "step_finished"}:
            key = event["step_id"]
            if event_type == "step_started" and self.steps.get(key, {}).get("status") in {"completed", "failed", "interrupted"}:
                return
            self.steps[key] = {**self.steps.get(key, {}), **event}
        elif event_type in {"session_started", "session_finished"}:
            key = event["session_id"]
            if event_type == "session_started" and self.sessions.get(key, {}).get("event") == "session_finished":
                return
            self.sessions[key] = {**self.sessions.get(key, {}), **event}

    def record(self, event):
        event = {"kind": "token_usage", "session_id": self.session_id, **event}
        with self.lock:
            self._apply(event)
            if self.persist:
                try:
                    self.persist(event)
                except Exception:
                    # 记账落盘失败不能触发模型重试、从而制造新的消耗。
                    self.persistence_errors += 1
            self.queue.put(event)
        if self.observer:
            self.observer(event)
        notify = active_notify.get()
        if notify and event.get("event") in {"step_started", "call_finished", "step_finished"}:
            # 只发送唤醒信号；终端由主线程消费队列，避免多个 Agent 输出交错。
            try:
                notify({"kind": "token_usage_ready"})
            except Exception:
                pass  # 流已关闭仍保留队列和落盘记录，不因显示失败触发推理重试。

    def observe(self, event):
        """进程总账只接收新事件；历史重放不再累计，不重复打印/持久化。"""
        with self.lock:
            self._apply(event)

    def begin_call(self, *, call_id=None, step=None, operation="模型调用", **details):
        call_id = str(call_id or uuid.uuid4().hex)
        scope = step or active_step.get() or {"step": "unscoped", "step_id": ""}
        self.record({"event": "call_started", "call_id": call_id, **scope,
                     "operation": operation, "status": "pending", **(active_task.get() or {}), **details})
        return call_id

    def finish_call(self, call_id, *, metadata=None, usage=None, failed=False, elapsed=None):
        with self.lock:
            if self.calls.get(str(call_id), {}).get("status") != "pending":
                return
            self.record({**self.calls[str(call_id)], "event": "call_finished",
                         "status": "failed" if failed or (metadata or {}).get("done_reason") == "length" else "completed", "elapsed": elapsed,
                         "output_truncated": (metadata or {}).get("done_reason") == "length",
                         **measured_usage(metadata, usage)})

    def report(self):
        with self.lock:
            current = [c for c in self.calls.values() if c["session_id"] == self.session_id]
            steps = []
            for step in self.steps.values():
                if step["session_id"] == self.session_id:
                    steps.append({**step, **summarize(c for c in current if c.get("step_id") == step["step_id"])})
            # 旧账本可能只有调用/节点事件；不能把缺失的整次执行时长当成零。
            sessions = dict(self.sessions)
            for event in [*self.calls.values(), *self.steps.values()]:
                sessions.setdefault(event["session_id"], {"session_id": event["session_id"]})
            timing = {
                "current": summarize_timing([s for sid, s in sessions.items() if sid == self.session_id], current),
                "cumulative": summarize_timing(sessions.values(), self.calls.values()),
                "steps": [{"step": s["step"], "step_id": s["step_id"], "session_id": s["session_id"],
                           "status": s["status"], "elapsed_seconds": _seconds(s.get("elapsed"))} for s in self.steps.values()],
            }
            timing["cumulative"]["historical_gap"] = self.historical_gap
            if self.historical_gap:
                timing["cumulative"]["complete"] = False
            return {"session_id": self.session_id, "current": summarize(current),
                    "cumulative": summarize(self.calls.values()), "steps": steps,
                    "timing": timing,
                    "has_history": any(sid != self.session_id for sid in sessions),
                    "historical_gap": self.historical_gap, "persistence_errors": self.persistence_errors}

    def drain(self, printer):
        while True:
            try:
                printer(self.queue.get_nowait())
            except queue.Empty:
                return


@contextmanager
def usage_session(ledger, *, scope="question"):
    token = active_usage.set(ledger)
    started = perf_counter()
    status = "failed"
    try:
        ledger.record({"event": "session_started", "historical_gap": ledger.historical_gap,
                       "scope": scope, "status": "running", "timing_version": 1, "started_at": time.time()})
        yield ledger
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        ledger.record({"event": "session_finished", "scope": scope, "status": status,
                       "timing_version": 1, "elapsed": round(perf_counter() - started, 3), "finished_at": time.time()})
        active_usage.reset(token)


def meter_node(name, function):
    """不修改 GraphState；每次节点执行单独记账，零调用步骤也可见。"""
    def measured(state):
        ledger = active_usage.get()
        if ledger is None:
            return function(state)
        started = perf_counter()
        step = {"step": name, "step_id": uuid.uuid4().hex}
        token = active_step.set(step)
        from langgraph.config import get_stream_writer
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = None
        notification = active_notify.set(writer)
        ledger.record({"event": "step_started", **step, "status": "running", "timing_version": 1})
        status = "failed"
        try:
            result = function(state)
            status = "completed"
            return result
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            active_step.reset(token)
            with ledger.lock:
                counts = summarize(c for c in ledger.calls.values() if c.get("step_id") == step["step_id"])
            ledger.record({"event": "step_finished", **step, "status": status, "usage": counts,
                           "timing_version": 1, "elapsed": round(perf_counter() - started, 3)})
            active_notify.reset(notification)
    measured.__name__ = name
    return measured


def _characters(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(c.get("text", "")) for c in content if isinstance(c, dict) and isinstance(c.get("text"), str))
    return 0


class UsageCallback(BaseCallbackHandler):
    """在解析器丢弃 AIMessage 之前读取用量；不保存提示、回答或思考正文。"""

    run_inline = True

    def __init__(self, ledger, *, operation, thinking_requested):
        self.ledger = ledger
        self.operation = operation
        self.thinking_requested = thinking_requested
        self.started = {}
        self.lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        roles = Counter()
        for batch in messages:
            for message in batch:
                roles[message.type] += _characters(message.content)
        with self.lock:
            self.started[str(run_id)] = time.monotonic()
        self.ledger.begin_call(call_id=run_id, operation=self.operation,
            model=(kwargs.get("invocation_params") or {}).get("model", ""),
            input_characters=dict(roles), thinking_requested=self.thinking_requested)

    def _finish(self, run_id, response=None, failed=False):
        metadata, usage = {}, {}
        # 每次链 invoke 对应一次 chat 请求；只取第一个候选，防止候选共享计数重复累计。
        if response and response.generations and response.generations[0]:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            metadata = {**(generation.generation_info or {}), **(getattr(message, "response_metadata", None) or {})}
            usage = getattr(message, "usage_metadata", None) or {}
        with self.lock:
            started = self.started.pop(str(run_id), None)
        self.ledger.finish_call(str(run_id), metadata=metadata, usage=usage, failed=failed,
                                elapsed=round(time.monotonic() - started, 3) if started is not None else None)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._finish(run_id, response)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, kwargs.get("response"), failed=True)


def _number(value):
    return "未报告" if value is None else f"{value:,}"


def print_usage_event(event):
    if event.get("event") == "step_finished":
        stats = event["usage"]
        label = STEP_LABELS.get(event["step"], event["step"])
        note = "；未调用对话模型" if not stats["calls"] else ""
        if not stats["complete"]:
            note += f"；{stats['unknown']} 次计数不完整，不能视为完整总量"
        print(f"  [步骤 Token · {label}] 调用 {stats['calls']} 次；输入 {stats['input_tokens']:,} + 生成 {stats['output_tokens']:,} = 已知合计 {stats['total_tokens']:,}{note}", flush=True)
        print(f"  [步骤耗时 · {label}] {format_duration(event.get('elapsed'))}", flush=True)
        return
    if event.get("event") != "call_finished":
        return
    task = f" · 任务 {event['task_id'][:12]} / 尝试 {event.get('task_attempt', 1)}" if event.get("task_id") else ""
    label = STEP_LABELS.get(event.get("step"), event.get("step", "模型调用"))
    print(f"  [Token · {label} · {event.get('operation', '')}{task} · {event['call_id'][:8]}]", flush=True)
    print(f"    输入：{_number(event.get('input_tokens'))}；生成总量：{_number(event.get('output_tokens'))}")
    print(f"    模型请求耗时：{format_duration(event.get('elapsed'))}（不含本地槽位排队）")
    if event.get("reasoning_tokens") is not None:
        print(f"    生成内部分项：思考 {_number(event['reasoning_tokens'])}；可见输出 {_number(event['visible_output_tokens'])}（不重复加总）")
    else:
        print("    思考 / 可见输出：服务未单列，无法精确拆分" + ("（已请求关闭思考）" if event.get("thinking_requested") is False else ""))
    if event.get("cached_input_tokens") is not None:
        print(f"    输入中缓存读取：{event['cached_input_tokens']:,}（输入的子集，不重复加总）")
    if event.get("input_characters"):
        roles = {"system": "系统消息（可能含注入证据）", "human": "用户消息（问题/资料）", "ai": "前轮模型结果", "tool": "工具结果"}
        print("    输入组成（字符，非 token）：" + "；".join(f"{roles.get(k, k)} {v:,}" for k, v in event["input_characters"].items()))
    if event.get("status") == "failed":
        print("    调用失败；已报告用量仍计入，未报告部分不当作零。")
    if event.get("output_truncated"):
        print("    输出额度耗尽：已拒绝截断结果，不原样重试；请调整生成/上下文额度。")


def print_compact_usage_event(event):
    """默认终端只输出步骤数字；逐调用明细留给详细模式和持久账本。"""
    label = STEP_LABELS.get(event.get("step"), event.get("step", "模型"))
    if event.get("event") == "step_started":
        print(f"[开始] {label}", flush=True)
    elif event.get("event") == "step_finished":
        stats = event["usage"]
        status = {"completed": "完成", "failed": "失败", "interrupted": "中断"}.get(event.get("status"), "未知")
        unknown = f" | 未知用量 {stats['unknown']} 次" if stats["unknown"] else ""
        print(f"[{status}] {label} | {format_duration(event.get('elapsed'))} | 调用 {stats['calls']} | "
              f"输入 {stats['input_tokens']:,} + 生成 {stats['output_tokens']:,} = {stats['total_tokens']:,} token{unknown}", flush=True)
    elif event.get("event") == "call_finished" and event.get("status") == "failed":
        reason = "生成截断" if event.get("output_truncated") else "请求失败"
        print(f"[模型异常] {label} | {reason} | {format_duration(event.get('elapsed'))} | "
              f"输入 {_number(event.get('input_tokens'))} / 生成 {_number(event.get('output_tokens'))}", flush=True)


def print_usage_summary(ledger, *, title="本次问题 Token 统计", session_ledger=None, compact=False):
    if compact:
        return print_compact_usage_summary(ledger, session_ledger=session_ledger)
    ledger.drain(print_usage_event)
    report = ledger.report()
    print(f"\n================== {title} ==================")
    for step in report["steps"]:
        label = STEP_LABELS.get(step["step"], step["step"])
        suffix = "；未调用对话模型" if not step["calls"] else ""
        if step["unknown"]:
            suffix += f"；{step['unknown']} 次用量不完整"
        print(f"  {label}：耗时 {format_duration(step.get('elapsed'))}；调用 {step['calls']} 次，输入 {step['input_tokens']:,}，生成 {step['output_tokens']:,}，已知合计 {step['total_tokens']:,}{suffix}")
    def total(label, stats):
        qualifier = "总计" if stats["complete"] else "已知小计（不是完整总量）"
        print(f"  {label}：输入 {stats['input_tokens']:,} + 生成 {stats['output_tokens']:,} = {qualifier} {stats['total_tokens']:,} token；调用 {stats['calls']} 次")
        if stats["unknown"] or stats["pending"]:
            print(f"    用量不完整 {stats['unknown']} 次；未返回 {stats['pending']} 次；失败 {stats['failed']} 次。不能补算未报告消耗。")
        if stats["calls"]:
            if stats["reasoning_unknown"]:
                print(f"    思考/可见输出未完整拆分：{stats['reasoning_unknown']} 次；生成总量已计入，不另外相加。")
            else:
                print(f"    生成内部分项：思考 {stats['reasoning_tokens']:,}；可见输出 {stats['visible_output_tokens']:,}。")
    total("本次执行新增", report["current"])
    if report["has_history"]:
        total("该研究历史累计（含本次）", report["cumulative"])
    if report["historical_gap"]:
        print("  提醒：旧研究没有历史 token 记录，无法还原升级前的消耗。")
    if report["persistence_errors"]:
        print(f"  警告：{report['persistence_errors']} 条计量事件未能落盘，历史累计可能不完整。")
    if report["cumulative"]["pending"]:
        print("  未返回调用可能仍在后台退出；研究模式可稍后用 --status RUN_ID 查看已落盘用量。")
    print("  口径：模型服务报告的输入 + 生成；包含内部 JSON/评估/重试，不仅是最终答案。")
    print("  预热单列；本地嵌入、搜索和数据库计算不计入对话模型 token，不代表没有计算开销。")
    print_timing_summary(ledger, report, session_ledger=session_ledger)
    print("\n------------------ Token 最终合计 ------------------")
    total("本次执行全部步骤", report["current"])
    if report["has_history"]:
        total("本研究累计（含恢复前已记录调用）", report["cumulative"])
    if session_ledger is not None:
        total("本次程序累计（含预热和本轮之前的问题，不重复计入历史）", session_ledger.report()["cumulative"])
    print("---------------------------------------------------", flush=True)


def print_timing_summary(ledger, report=None, *, session_ledger=None):
    report = report or ledger.report()
    print("\n------------------ 耗时统计 ------------------")
    def duration(label, stats):
        suffix = f"（已知小计；{stats['unknown_sessions']} 次执行缺少完整计时）" if not stats["complete"] else ""
        if stats.get("historical_gap"):
            suffix = "（已知小计；升级前的历史执行时间未记录，无法补算）"
        print(f"  {label}：{format_duration(stats['elapsed_seconds'])}{suffix}")
    scope = ledger.sessions.get(ledger.session_id, {}).get("scope")
    duration("本次预热耗时" if scope == "warmup" else "本次执行总耗时（不含预热与等待输入）", report["timing"]["current"])
    if report["has_history"]:
        duration("本研究累计执行耗时（不含中断间隔）", report["timing"]["cumulative"])
    if session_ledger is not None:
        duration("本程序累计执行耗时（含预热，不含等待输入）", session_ledger.report()["timing"]["cumulative"])
    stats = report["timing"]["current"]
    suffix = f"；{stats['unknown_model_requests']} 次未记录" if stats["unknown_model_requests"] else ""
    print(f"  本次模型请求耗时之和：{format_duration(stats['model_request_seconds'])}{suffix}（并发可能重叠，不是总耗时）")
    print("  步骤耗时包含检索、排队、重试和本地处理；模型请求耗时不是纯GPU计算时间。")


def print_compact_usage_summary(ledger, *, session_ledger=None):
    ledger.drain(print_compact_usage_event)
    report = ledger.report()
    def line(label, stats, timing):
        qualifier = "总计" if stats["complete"] else "已知小计（不是完整总量）"
        elapsed = format_duration(timing['elapsed_seconds']) if timing['complete'] else f"已知 {format_duration(timing['elapsed_seconds'])}，历史/未完计时缺失"
        print(f"{label}：{elapsed} | 调用 {stats['calls']} | 输入 {stats['input_tokens']:,} + "
              f"生成 {stats['output_tokens']:,} = {qualifier} {stats['total_tokens']:,} token | "
              f"失败 {stats['failed']} / 未返回 {stats['pending']} / 未知用量 {stats['unknown']}")
    scope = ledger.sessions.get(ledger.session_id, {}).get("scope")
    print("\n[预热汇总]" if scope == "warmup" else "\n[Token 最终合计 / 耗时统计]", flush=True)
    line("预热" if scope == "warmup" else "本次", report['current'], report['timing']['current'])
    if report['has_history']:
        line("研究累计", report['cumulative'], report['timing']['cumulative'])
    if session_ledger is not None:
        process = session_ledger.report()
        line("程序累计（含预热）", process['cumulative'], process['timing']['cumulative'])
    stats = report['current']
    if stats['calls']:
        print("思考/可见输出：未单列；已包含在生成量中。" if stats['reasoning_unknown'] else
              f"生成内部分项：思考 {stats['reasoning_tokens']:,} / 可见输出 {stats['visible_output_tokens']:,}（不重复相加）")
    if report['historical_gap']:
        print("[提示] 旧研究的历史消耗未记录，累计不完整。")
    if report['persistence_errors']:
        print(f"[警告] {report['persistence_errors']} 条统计事件落盘失败，累计可能不完整。")
