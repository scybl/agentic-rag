"""只读研究诊断：区分节点结果已写入、图待推进和真正结束。"""

import json
import sqlite3
from collections import Counter
from pathlib import Path

from ..config import settings
from ..token_usage import UsageLedger, STEP_LABELS, format_duration


def has_pending_work(snapshot):
    # LangGraph 在结果写入、下一检查点提交之间可能 next=()，tasks仍带已完成节点结果。
    return bool(snapshot.next or getattr(snapshot, "tasks", ()))


def read_snapshot(run_id, path=None):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from ..graph.build import build_graph
    path = Path(path or settings.checkpoint_db).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, check_same_thread=False) as connection:
        saver = SqliteSaver(connection)
        # 仅查看已经存在的表，跳过 setup 的DDL/PRAGMA写入；缺表由调用方报告。
        saver.is_setup = True
        return build_graph(checkpointer=saver).get_state({"configurable": {"thread_id": run_id}})


def inspect_research(database, run_id, *, snapshot=None):
    run = database.run_info(run_id)
    if run is None:
        raise ValueError("没有找到该研究编号")
    events = database.events(run_id, limit=-1)
    ledger = UsageLedger(history=[e for e in events if e.get("kind") == "token_usage"])
    tasks = database.tasks(run_id)
    state, checkpoint_error, pending, staged = {}, "", [], []
    try:
        snapshot = snapshot if snapshot is not None else read_snapshot(run_id)
        state = dict(snapshot.values)
        pending = list(snapshot.next)
        staged = [task.name for task in getattr(snapshot, "tasks", ()) if getattr(task, "result", None) is not None]
        unfinished = has_pending_work(snapshot)
    except Exception as exc:
        checkpoint_error = f"检查点不可读（{type(exc).__name__}）：{exc}"
        unfinished = None
    steps = list(ledger.steps.values())
    failure = next((e for e in reversed(events) if e.get("kind") == "run_failure"), None)
    if failure is None and run["status"] == "interrupted":
        failure = {"error_type": "KeyboardInterrupt", "message": "旧记录显示程序捕获中断；无法确定由键盘、终端停止还是外部信号触发。"}
    tool_calls = {e.get("tool_call_id") for e in events if e.get("kind") == "tool" and e.get("phase") == "started"}
    reading = state.get("reading_reports", [])
    return {
        "run": run, "token_usage": ledger.report(), "last_step": steps[-1] if steps else None,
        "pending_nodes": pending, "staged_nodes": staged, "has_pending_work": unfinished,
        "checkpoint_error": checkpoint_error, "failure": failure,
        "counts": {"tool_calls": len(tool_calls), "tasks": dict(Counter(t['status'] for t in tasks)),
                   "evidence": len(state.get("documents", [])), "articles_read": len(reading),
                   "segments_read": sum(r.get("covered", 0) for r in reading),
                   "segments_total": sum(r.get("total", 0) for r in reading)},
        "has_answer": bool(state.get("generation")),
        "answer_verified": bool(state.get("generation_grounded") and state.get("generation_complete")),
        "analysis": {key: state.get(key) for key in (
            "selected_sources", "source_queries", "news_search_plan", "task_type", "evidence_needs", "plan_summary",
            "evidence_assessment", "answer_review", "source_errors", "reading_reports", "specialist_findings")},
        "news_searches": [e for e in events if e.get("kind") == "news_ranked"],
        "tasks": tasks, "events": events,
    }


def print_inspection(report, *, verbose=False, include_usage=True):
    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return
    run, counts = report['run'], report['counts']
    print(f"[研究] {run['id']} | 状态 {run['status']}")
    print(f"问题：{run['question']}")
    if include_usage:
        usage = report['token_usage']['cumulative']
        timing = report['token_usage']['timing']['cumulative']
        duration = format_duration(timing['elapsed_seconds']) + ("（已知小计）" if not timing['complete'] else "")
        qualifier = "总计" if usage['complete'] else "已知小计"
        print(f"耗时 {duration} | 调用 {usage['calls']} | 输入 {usage['input_tokens']:,} + 生成 {usage['output_tokens']:,} "
              f"= {qualifier} {usage['total_tokens']:,} token | 失败 {usage['failed']} / 未返回 {usage['pending']}")
    step = report['last_step'] or {}
    print(f"最后步骤：{STEP_LABELS.get(step.get('step'), step.get('step', '未记录'))} / {step.get('status', '未知')} | "
          f"工具 {counts['tool_calls']} 次 | 证据 {counts['evidence']} 条 | 阅读 {counts['segments_read']}/{counts['segments_total']} 段 | "
          f"答案 {'已核验' if report['answer_verified'] else '未核验草稿' if report['has_answer'] else '未生成'}")
    if counts['tasks']:
        print("任务状态：" + " / ".join(f"{key} {value}" for key, value in counts['tasks'].items()))
    for search in report.get("news_searches", []):
        print(f"新闻候选：{search['candidate_count']} | 正文入选 {search['selected_count']} | "
              f"暂未精读 {search.get('deferred_count', '未知')} | "
              f"分页 {'已取完' if search.get('retrieval_complete') else '未取完/未知'}；完整候选与权重见 -v 的 news_candidates")
    failed_tasks = [task for task in report['tasks'] if task.get('error')]
    for task in failed_tasks[:3]:
        print(f"任务异常 {task['key'][:12]}：{task['error']}")
    if len(failed_tasks) > 3:
        print(f"其余 {len(failed_tasks)-3} 个任务异常见 -v。")
    assessment = report['analysis'].get('evidence_assessment') or {}
    if assessment:
        print(f"证据检查：可用 {len(assessment.get('usable_evidence_ids', []))} / 缺口 {len(assessment.get('missing_factors', []))} / 风险 {len(assessment.get('concerns', []))}")
    for source, error in (report['analysis'].get('source_errors') or {}).items():
        print(f"来源异常/{source}：{error}")
    if report['failure']:
        print(f"停止原因：{report['failure'].get('error_type')} | {report['failure'].get('message')}")
    if report['checkpoint_error']:
        print(f"[诊断提示] {report['checkpoint_error']}")
    if report['pending_nodes']:
        print("待执行：" + ", ".join(report['pending_nodes']))
    if report['staged_nodes']:
        print("已保存节点结果、等待流程推进：" + ", ".join(report['staged_nodes']))
    for source, query in (report['analysis'].get('source_queries') or {}).items():
        print(f"计划查询（不代表已执行）/{source}：{query}")
    plan = report['analysis'].get('news_search_plan') or {}
    if plan:
        print(f"新闻日期：{plan.get('start') or '不限'} ~ {plan.get('end') or '不限'} | {plan.get('time_note', '')}")
        print(f"新闻精确时间：{plan.get('published_after') or '不限'} ~ {plan.get('published_before') or '不限'}")
        print(f"新闻约束：人物 {'、'.join(plan.get('people', [])) or '不限'} | 机构 {'、'.join(plan.get('organizations', [])) or '不限'} | "
              f"主题 {'、'.join(plan.get('topics', [])) or '不限'} | 来源 {'、'.join(plan.get('source_names', [])) or '不限'} | "
              f"栏目 {plan.get('section') or '不限'} | 排序 {plan.get('sort_by', 'relevance')} | "
              f"覆盖 {plan.get('coverage', 'focused')} | 入选 {plan.get('result_limit', 5)}")
    if report['has_pending_work']:
        print(f"继续：agentic-rag --resume {run['id']}（启动时校验配置兼容性）")
    elif not report['has_answer'] and not report['checkpoint_error']:
        print("[诊断提示] 没有最终答案，也没有可推进任务；不能认定研究已完成。")
