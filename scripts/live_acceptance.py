"""新鲜取证的 CLI 实跑；不把固定资料重放伪装成联网端到端验证。"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_rag.telemetry.exporters import write_json
from agentic_rag.token_usage import UsageLedger
from agentic_rag.evaluation.acceptance import implementation


QUESTIONS = [
    "只使用新闻API，阅读最新2篇关于新疆美盛的新闻，说明临时停产原因、预计时长和恢复条件。不要网页搜索。",
    "只使用新闻API，阅读最新3篇关于AI短剧的新闻，分析发展前景和盈利风险。事实与有条件推断分开，不编造未来规模。不要网页搜索。",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--news-base-url", default="")
    parser.add_argument("--transport-note", default="configured public endpoint")
    parser.add_argument("--seconds", type=int, default=480)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "RESEARCH_DB": str(output / "research.sqlite3"),
           "CHECKPOINT_DB": str(output / "checkpoints.sqlite3"), "MEMORY_VECTOR_DIR": str(output / "memory"),
           "ANALYSIS_CACHE_ENABLED": "false", "RESEARCH_TOTAL_TIMEOUT": str(args.seconds),
           "RESEARCH_MAX_MODEL_CALLS": "40", "OLLAMA_KEEP_ALIVE": "30m"}
    if args.news_base_url:
        env["NEWS_API_BASE_URL"] = args.news_base_url
    source = implementation()
    write_json(output / "manifest.json", {"scope": "fresh-news CLI end-to-end", "questions": QUESTIONS,
        "implementation": source,
        "transport_note": args.transport_note, "seconds_per_case": args.seconds, "max_calls": 40,
        "started_at": time.time(), "analysis_cache": False,
        "note": "每题 CLI 预热/卸载照常执行，日志保留。研究库/检查点隔离，已有新闻摘要向量可能复用。"})
    results = []
    for i, question in enumerate(QUESTIONS, 1):
        if implementation() != source:
            raise RuntimeError("源码在实跑期间变化，停止混合版本验收")
        started = time.perf_counter()
        with (output / f"live-{i}.log").open("w", encoding="utf-8") as log:
            try:
                child = subprocess.run([sys.executable, "-X", "utf8", "-m", "agentic_rag.cli", question],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds + 90, check=False)
                code = child.returncode
            except subprocess.TimeoutExpired:
                code = "wall_timeout"
        entry = {"question": question, "process_exit": code, "process_seconds": time.perf_counter()-started,
                 "implementation_unchanged": implementation() == source,
                 "scope": "fresh-news end-to-end", "transport_note": args.transport_note}
        database = output / "research.sqlite3"
        if database.exists():
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
                row = db.execute("SELECT id,status FROM runs WHERE question=? ORDER BY created DESC LIMIT 1", (question,)).fetchone()
                if row:
                    records = [json.loads(r[0]) for r in db.execute("SELECT data FROM events WHERE run_id=? ORDER BY id", (row[0],))]
                    entry.update(run_id=row[0], status=row[1], usage=UsageLedger(history=records).report()["cumulative"],
                        news_requests=sum(e.get("kind") == "news_request" for e in records),
                        reading_completed=sum(e.get("event") == "completed" and e.get("role") == "reader" for e in records),
                        specialist_completed=sum(e.get("event") == "completed" and e.get("role") == "specialist" for e in records))
                    checkpoint = output / "checkpoints.sqlite3"
                    if checkpoint.exists():
                        with sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True) as conn:
                            saved = SqliteSaver(conn).get_tuple({"configurable": {"thread_id": row[0]}})
                        if saved:
                            state = saved.checkpoint["channel_values"]
                            docs = state.get("answer_documents", [])
                            entry.update(selected_sources=state.get("selected_sources", []),
                                source_errors=state.get("source_errors", {}),
                                generation_complete=state.get("generation_complete", False),
                                generation_grounded=state.get("generation_grounded", False),
                                answer=state.get("generation", ""),
                                execution_violations=state.get("execution_violations", []),
                                reading_reports=state.get("reading_reports", []),
                                specialist_execution=state.get("specialist_execution", {}),
                                used_article_ids=[d.metadata.get("article_id") for d in docs if d.metadata.get("source_type") == "news_api"])
        entry["passed"] = (entry["implementation_unchanged"] and code == 0 and entry.get("status") == "completed"
            and entry.get("generation_complete") and entry.get("generation_grounded")
            and entry.get("selected_sources") == ["news_api"] and not entry.get("source_errors")
            and len(set(entry.get("used_article_ids", []))) == (2 if i == 1 else 3))
        results.append(entry)
        write_json(output / "results.json", results)
        print(json.dumps({k: entry[k] for k in ("question", "process_exit", "process_seconds", "passed", "transport_note")}, ensure_ascii=False), flush=True)
        if not entry["implementation_unchanged"]:
            raise RuntimeError("当前题目执行期间源码发生变化；已保留结果，拒绝标为通过")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
