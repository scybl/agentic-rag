"""隔离实机验收：恢复旧阅读故障，或从真实 API 开始一轮新研究；不修改原研究库。"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("resume", "fresh", "retrieval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="d2bdb6d7899f40d7b6cdd2ce4f52b6d4")
    parser.add_argument("--question", default="只使用新闻API，分析厄尔尼诺对农业和能源的直接影响及后续风险，不限制新闻日期，不要网页搜索。")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        from agentic_rag.config import settings
        output.mkdir(parents=True, exist_ok=False)
        if args.mode == "resume":
            for source, name in ((settings.research_db, "research.sqlite3"), (settings.checkpoint_db, "checkpoints.sqlite3")):
                with sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True) as original:
                    with sqlite3.connect(output / name) as destination:
                        original.backup(destination)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "RESEARCH_DB": str(output / "research.sqlite3"),
               "CHECKPOINT_DB": str(output / "checkpoints.sqlite3"), "MEMORY_VECTOR_DIR": str(output / "memory"),
               "ANALYSIS_CACHE_PATH": str(output / "answers.sqlite3"), "OLLAMA_KEEP_ALIVE": "30m"}
        with (output / "execution.log").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-X", "utf8", __file__, *sys.argv[1:], "--worker"],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        result_path = output / "result.json"
        print(result_path.read_text(encoding="utf-8") if result_path.exists() else f"验收异常退出：{process.returncode}，日志：{output / 'execution.log'}")
        return process.returncode

    import uuid
    from langgraph.checkpoint.sqlite import SqliteSaver
    from agentic_rag.cli import ask
    from agentic_rag.config import settings
    from agentic_rag.graph.build import build_graph
    from agentic_rag.research.service import store
    from agentic_rag.evaluation.acceptance import implementation
    from agentic_rag.token_usage import UsageLedger

    if args.mode == "retrieval":
        return verify_retrieval(args, output)

    run_id = args.run_id if args.mode == "resume" else uuid.uuid4().hex
    source = implementation()
    started = time.perf_counter()
    db = store()
    old_tasks = {t["key"]: t for t in db.tasks(run_id)}
    old_events = len(db.events(run_id, limit=-1))
    with SqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        passed = ask(graph, args.question if args.mode == "fresh" else "", run_id=run_id, resume=args.mode == "resume")
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
    state = dict(snapshot.values)
    events = db.events(run_id, limit=-1)[old_events:]
    reports = state.get("reading_reports", [])
    result = {"scope": "真实模型 + 旧原文恢复" if args.mode == "resume" else "真实API + 真实模型 + 新研究",
              "run_id": run_id, "elapsed": round(time.perf_counter() - started, 3),
              "passed": passed and implementation() == source, "implementation": source,
              "implementation_unchanged": implementation() == source, "pending": list(snapshot.next),
              "generation_complete": state.get("generation_complete"), "generation_grounded": state.get("generation_grounded"),
              "source_errors": state.get("source_errors", {}), "violations": state.get("execution_violations", []),
              "reading": {"articles": len(reports), "covered": sum(r["covered"] for r in reports),
                          "total": sum(r["total"] for r in reports)},
              "tasks_before": len(old_tasks), "tasks_after": len(db.tasks(run_id)),
              "reused": sum(e.get("event") in {"reused", "article_reused"} for e in events),
              "reader_completed": sum(e.get("event") == "completed" and e.get("role") == "reader" for e in events),
              "news_requests": sum(e.get("kind") == "news_request" for e in events),
              "screens": [e for e in events if e.get("kind") in {"news_screen_progress", "news_core_ready"}],
              "specialists": state.get("specialist_execution", {}),
              "usage": UsageLedger(history=events).report()["cumulative"], "answer": state.get("generation", "")}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


def verify_retrieval(args, output):
    """仅验收真实全量摘要筛选与核心正文选取，明确不冒充完整研究。"""
    import uuid
    from agentic_rag.budget import research_budget
    from agentic_rag.console import CompactTrace
    from agentic_rag.evaluation.acceptance import implementation
    from agentic_rag.news_screening import screen_news
    from agentic_rag.research.service import store
    from agentic_rag.token_usage import UsageLedger, meter_node, usage_session
    from agentic_rag.tools import search_news
    from agentic_rag.tools.contracts import ToolContext
    from agentic_rag.tools.execution import execute_tool
    db, run_id, trace = store(), uuid.uuid4().hex, CompactTrace()
    db.start_run(run_id, args.question)
    ledger = UsageLedger(persist=lambda e: db.event(run_id, e))
    def emit(event):
        db.event(run_id, event)
        trace.show_event(event)
    source, started = implementation(), time.perf_counter()
    result = {"scope": "真实API + 真实模型：摘要筛选/核心正文/摘要缓存复用；不含最终答案", "run_id": run_id}
    try:
        with usage_session(ledger), research_budget(seconds=1800, calls=160):
            response = meter_node("collect_sources", lambda _: execute_tool(search_news, {
                "semantic_query": args.question, "queries": ["厄尔尼诺"], "result_limit": 12,
            }, context=ToolContext(emit=emit, caller="实机验收", reason="全分页摘要筛选后选择12篇核心正文")))({})
            bundle = response.artifact
            previous = len(ledger.calls)
            repeated = meter_node("collect_sources", lambda _: screen_news(bundle.candidates, args.question, emit=emit))({})
            repeated_calls = len(ledger.calls) - previous
            events = db.events(run_id, limit=-1)
            ready = next((e for e in reversed(events) if e.get("kind") == "news_core_ready"), {})
            result.update(candidates=len(bundle.candidates), core=len(bundle.documents), warnings=bundle.warnings,
                          retrieval_complete=bundle.retrieval_complete, core_summary=ready,
                          article_ids=[d.metadata.get("article_id") for d in bundle.documents],
                          news_requests=sum(e.get("kind") == "news_request" for e in events),
                          screens=[e for e in events if e.get("kind") == "news_screen_progress"],
                          repeat_count=len(repeated), repeat_model_calls=repeated_calls,
                          passed=bool(bundle.retrieval_complete and len(bundle.documents) == 12 and not bundle.warnings
                                      and repeated_calls == 0 and all(d.metadata.get("content_kind") == "news_article" for d in bundle.documents)))
    except Exception as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    result.update(elapsed=round(time.perf_counter()-started, 3), usage=ledger.report()["cumulative"],
                  implementation=source, implementation_unchanged=implementation() == source)
    result["passed"] = result["passed"] and result["implementation_unchanged"]
    db.run_status(run_id, "completed" if result["passed"] else "failed")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
