"""把既有事件转换为稳定父子 Trace；不把丢失时间或用量补成零。"""

import math

from .schema import Event, Span, Trace, identity


# 有意不导出 question、prompt、正文、模型输出、工具参数、URL、错误消息和 goal。
SAFE_FIELDS = {
    "session_id", "step_id", "task_id", "task_key", "task_attempt", "call_id",
    "tool_call_id", "checkpoint_id", "step", "task_kind", "role", "model", "model_revision",
    "workflow_revision", "reading_recipe", "prompt_version", "operation", "scope",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens",
    "visible_output_tokens", "thinking_requested", "output_truncated", "historical_gap",
    "count", "offset", "candidate_count", "selected_count", "deferred_count", "raw_count",
    "ranking_method", "vector_cache_used", "retrieval_complete", "retry_skipped",
    "error_type", "status", "event", "phase", "active", "limit", "rule", "passed",
    "timing_version", "elapsed", "at", "started_at", "finished_at",
}


def number(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value
    return None


def score(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def public_fields(data):
    fields = {key: value for key, value in data.items() if key in SAFE_FIELDS
              and (value is None or isinstance(value, (str, bool)) or number(value) is not None)}
    if data.get("task_key"):
        fields["task_id"] = str(data["task_key"])
    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "visible_output_tokens", "cached_input_tokens"):
        if key in data:
            value = data[key]
            fields[key] = value if type(value) is int and value >= 0 else None
    return fields


def candidates(data):
    """保留排名分解与入选标记；哈希候选标识以免 URL 中的参数随报告外泄。"""
    result = []
    for item in data.get("candidates", []):
        priority = item.get("priority") or {}
        key = str(item.get("candidate_id") or item.get("article_id") or "")
        result.append({
            "document_ref": identity("document", key),
            "selected": item.get("selected_for_reading") if isinstance(item.get("selected_for_reading"), bool) else None,
            "rank": number(priority.get("rank")), "score": score(priority.get("score")),
            "components": {k: v for k, v in (priority.get("components") or {}).items()
                           if k in {"semantic", "lexical", "query_coverage"} and score(v) is not None},
            "weights": {k: v for k, v in (priority.get("weights") or {}).items()
                        if k in {"semantic", "lexical", "query_coverage"} and number(v) is not None},
        })
    return result


def research_trace(run, records) -> Trace:
    run_id = str(run["id"])
    root = identity(run_id, "run")
    spans = {root: Span(span_id=root, kind="run", name="research",
                       status={"completed": "completed", "failed": "failed", "interrupted": "interrupted",
                               "needs_attention": "failed", "running": "pending"}.get(run.get("status"), "unknown"),
                       started_at=number(run.get("created")),
                       attributes={"question_hash": identity(str(run.get("question", ""))),
                                   "duration_scope": "session durations exclude resume gaps"})}
    events, seen, warnings = [], {}, set()

    def ensure(key, parent, name, kind):
        sid = identity(run_id, *key)
        if sid not in spans:
            spans[sid] = Span(span_id=sid, parent_id=parent, name=name, kind=kind)
        return sid

    def lineage(data):
        parent = root
        session = str(data.get("session_id") or "legacy")
        if data.get("session_id"):
            parent = ensure(("session", session), root, "execution", "session")
        if data.get("step_id"):
            parent = ensure(("node", session, str(data["step_id"])), parent,
                            str(data.get("step") or "legacy node"), "node")
        return parent, session

    def task_parent(data, parent, session):
        task = data.get("task_key") or data.get("task_id")
        attempt = data.get("task_attempt")
        if not task or attempt is None:
            return parent
        sid = ensure(("task", session, str(data.get("step_id", "")), str(task), str(attempt)),
                     parent, str(data.get("role") or data.get("task_kind") or "task"), "task")
        spans[sid].attributes.update(task_id=str(task), attempt=attempt)
        return sid

    def update(sid, data, at, beginning, ending, status=None):
        span = spans[sid]
        # 延迟到达的 started 可以补起点，但不能把完成态属性和用量退回 pending。
        fields = public_fields(data)
        if beginning and span.status in {"completed", "failed", "interrupted"}:
            for key, value in fields.items():
                span.attributes.setdefault(key, value)
        else:
            span.attributes.update(fields)
        if beginning:
            recorded = number(data.get("started_at"))
            span.started_at = at if recorded is None else recorded
            if span.status == "unknown":
                span.status = "pending"
        if ending:
            recorded = number(data.get("finished_at"))
            span.finished_at = at if recorded is None else recorded
            span.elapsed_seconds = number(data.get("elapsed"))
            span.status = status or {"completed": "completed", "failed": "failed",
                                     "interrupted": "interrupted"}.get(data.get("status"), "unknown")
        if at is not None and data.get("at") is None:
            span.attributes["timestamp_origin"] = "database_observed_legacy"
        else:
            span.attributes["timestamp_origin"] = "event_recorded"

    # 持久事件 ID 定义重放顺序，不按墙钟排序（系统时钟可能回拨）。
    for row in sorted(records, key=lambda r: r["id"]):
        eid = identity(run_id, "event", str(row["id"]))
        if eid in seen:
            if row != seen[eid]:
                raise ValueError("同一持久事件 ID 出现互相冲突的记录")
            continue
        seen[eid] = row
        data = row["data"]
        kind, name = data.get("kind", "unknown"), data.get("event") or data.get("phase") or data.get("kind", "unknown")
        at = number(data.get("at"))
        if at is None:
            at = number(row.get("at"))
            warnings.add("旧记录仅有数据库接收时间；不据此倒算缺失的耗时。")
        parent, session = lineage(data)
        target = parent
        if kind == "token_usage":
            if name.startswith("session_"):
                target = ensure(("session", session), root, "execution", "session")
            elif name.startswith("call_"):
                parent = task_parent(data, parent, session)
                target = ensure(("llm", str(data["call_id"])), parent, str(data.get("operation") or "model"), "llm")
            update(target, data, at, name.endswith("_started"), name.endswith("_finished"))
        elif kind == "research":
            target = task_parent(data, parent, session)
            if target != parent:
                update(target, data, at, name == "started", name in {"completed", "attempt_failed"},
                       "completed" if name == "completed" else "failed")
        elif kind == "tool" and data.get("tool_call_id"):
            parent = task_parent(data, parent, session)
            target = ensure(("tool", str(data["tool_call_id"])), parent, str(data.get("tool") or "tool"), "tool")
            update(target, data, at, name == "started", name in {"finished", "failed"},
                   "failed" if name == "failed" or data.get("status") == "error" else "completed")
        fields = public_fields(data)
        if kind == "news_candidates":
            fields["candidates"] = candidates(data)
        events.append(Event(event_id=eid, span_id=target, name=f"{kind}.{name}", at=at, attributes=fields))
    if not records:
        warnings.add("没有持久事件，无法还原历史用量或调用树。")
    if any(s.status == "pending" for s in spans.values() if s.kind != "run"):
        warnings.add("存在开始后未收到结束事件的操作；不能视为成功或零消耗。")
    return Trace(run_id=run_id, spans=list(spans.values()), events=events, warnings=sorted(warnings))
