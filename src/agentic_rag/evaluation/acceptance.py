"""真实材料/真实模型验收：冻结来源对照与新鲜联网运行分别留档。

python -m agentic_rag.evaluation.acceptance freeze --output evaluation/runs/news-frozen
python -m agentic_rag.evaluation.acceptance run --snapshot evaluation/runs/news-frozen/snapshot.json --output evaluation/runs/news-acceptance
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..telemetry.exporters import write_json

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "evaluation/suites/news_acceptance_v1/cases.json"


def implementation():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted((ROOT / "src/agentic_rag").rglob("*.py"))}


def configuration():
    """只记录复现实验所需的白名单，绝不序列化整个配置或认证字段。"""
    from ..config import settings
    names = ("llm_model", "llm_reasoning", "temperature", "llm_context_window", "llm_max_output_tokens",
             "llm_request_timeout", "llm_max_attempts", "llm_adaptive_max_attempts", "llm_adaptive_max_output_tokens",
             "llm_concurrency", "agent_workers", "reading_chunk_chars", "reading_verbatim_max_chars",
             "generation_context_chars", "generation_max_documents", "max_retries", "specialist_count", "embedding_model")
    return {name: getattr(settings, name) for name in names}


def freeze(output, database=None):
    from ..config import settings
    definition = json.loads(CASES.read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = Path(database or settings.research_db).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id,body,metadata FROM versions").fetchall()
    documents = {}
    for case in definition["cases"]:
        for prefix in case["sources"]:
            if prefix in documents:
                continue
            matches = [r for r in rows if r[0].startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"冻结来源必须唯一：{prefix}")
            version, body, metadata = matches[0]
            original = json.loads(metadata)
            # 排除以前研究的问题相关性、影响评分、模型意见，不能泄露先前答案。
            clean = {k: original[k] for k in ("source", "url", "source_type", "source_name", "title", "article_id",
                                             "published_at", "content_kind", "section") if k in original}
            documents[prefix] = {"version_id": version, "page_content": body, "metadata": clean,
                                 "sha256": hashlib.sha256(body.encode()).hexdigest()}
        source = "\n".join(documents[p]["page_content"] for p in case["sources"])
        if any(quote not in source for quote in case["quotes"]):
            raise ValueError(f"标注引文不在冻结原文：{case['id']}")
        for pattern in [*case["required"], *case["forbidden"]]:
            re.compile(pattern)
    snapshot = {**definition, "frozen_at": time.time(), "documents": documents,
                "label_policy": "基于原文编写并逐字核对的开发集，未获独立人工双审；正则仅是必要条件，不是独立事实/推理裁判"}
    write_json(output / "snapshot.json", snapshot)
    print(f"冻结 {len(documents)} 篇真实存档、{len(definition['cases'])} 个问题：{output}", flush=True)


def checks(case, state):
    answer = re.sub(r"\*\*|__", "", state.get("generation", ""))
    required = {p: bool(re.search(p, answer)) for p in case["required"]}
    forbidden = {p: bool(re.search(p, answer)) for p in case["forbidden"]}
    ids = set(re.findall(r"\bE\d+\b", answer))
    context_ids = set(re.findall(r"\[(E\d+)\]", state.get("evidence_context", "")))
    return {"required": required, "forbidden": forbidden,
            "valid_citations": bool(ids) and ids.issubset(context_ids),
            "checks_passed": all(required.values()) and not any(forbidden.values()) and bool(ids) and ids.issubset(context_ids)}


def worker(args):
    # 父进程为每一组设置隔离存储；这里不修改用户日常研究库。
    from langchain_core.documents import Document
    from langgraph.checkpoint.sqlite import SqliteSaver
    from ..budget import research_budget
    from ..config import settings
    from ..graph.build import build_graph
    from ..research import service
    from ..token_usage import UsageLedger, usage_session
    from ..tools.guardrails import infer_estimate_kind, is_probability_evidence_question

    output = Path(args.output)
    source_fingerprint = implementation()
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    case = next(c for c in snapshot["cases"] if c["id"] == args.case)
    run_id = uuid.uuid4().hex
    database = service.store()
    database.start_run(run_id, case["question"])
    events = []
    def persist(event):
        events.append(event)
        database.event(run_id, event)
    ledger = UsageLedger(persist=persist)
    result, error, signature, state = {}, None, None, {}
    started = time.perf_counter()
    try:
        with usage_session(ledger), research_budget(seconds=args.seconds, calls=args.calls):
            signature = service.model_revision()
            # 两组给同一批完整来源，排除搜索漂移；标签、答案与判分正则不传给模型。
            documents = [Document(page_content=snapshot["documents"][p]["page_content"],
                                  metadata=snapshot["documents"][p]["metadata"]) for p in case["sources"]]
            question = case["question"] + " 仅依据所给报道，简洁回答并引用[E编号]；数值按来源口径，不把报道当作独立核实。"
            kind = "forecast" if case["category"] in {"forecast", "probability"} else "analysis" if case["category"] in {"analysis", "conflict"} else "factual"
            if is_probability_evidence_question(case["question"]):
                kind = "factual"
            state = {"question": question, "original_question": question, "run_id": run_id,
                "documents": documents, "selected_sources": ["news_api"], "source_queries": {},
                "query_history": [], "source_errors": {}, "retries": 999, "no_progress_rounds": 2,
                "task_type": kind, "estimate_kind": infer_estimate_kind(case["question"], "directional" if kind == "forecast" else "none"),
                "evidence_needs": ["核对事实、数值、单位与事件时间", "区分来源事实、观点与条件推断"],
                "model_revision": signature, "workflow_revision": service.WORKFLOW_VERSION}
            if args.variant == "research":
                state["reading_recipe"] = service.recipe(signature)
            with SqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
                graph = build_graph(checkpointer=saver, research=args.variant == "research")
                config = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
                graph.update_state(config, state, as_node="collect_sources")
                result = graph.invoke(None, config=config)
        passed = bool(result.get("generation_complete") and result.get("generation_grounded"))
        database.run_status(run_id, "completed" if passed else "needs_attention")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        database.run_status(run_id, "failed")
    report = {"case_id": case["id"], "variant": args.variant, "run_id": run_id,
        "scope": "frozen-source-real-model; router/search/recall excluded; production graph resumes at grade_documents",
        "question": case["question"], "error": error, "elapsed_seconds": time.perf_counter() - started,
        "workflow_passed": bool(result.get("generation_complete") and result.get("generation_grounded")),
        "answer": result.get("generation", ""), "evidence_context": result.get("evidence_context", ""),
        "reading_reports": result.get("reading_reports", []), "specialist_execution": result.get("specialist_execution", {}),
        "answer_review": result.get("answer_review", {}), "assessment": result.get("evidence_assessment", {}),
        "usage": ledger.report(), "checks": checks(case, result), "implementation": source_fingerprint,
        "implementation_unchanged": source_fingerprint == implementation(),
        "model_revision": signature, "configuration": configuration(), "events": events,
        "planned_task_type": state.get("task_type"), "planned_estimate_kind": state.get("estimate_kind"),
        "manual_review": "pending"}
    write_json(output, report)
    print(json.dumps({k: report[k] for k in ("case_id", "variant", "run_id", "workflow_passed", "elapsed_seconds", "error")}, ensure_ascii=False), flush=True)
    return 0 if report["workflow_passed"] and report["checks"]["checks_passed"] and report["implementation_unchanged"] else 1


def summarize(output, expected):
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(output).glob("N*-*.json"))]
    result = {"expected": expected, "recorded": len(records), "missing": expected - len(records), "variants": {},
              "interpretation": "必要条件检查通过不代表答案质量已独立验证；失败、超时均保留分母；同源配对开发集不是线上延迟基准"}
    for variant in sorted({r["variant"] for r in records}):
        group = [r for r in records if r["variant"] == variant]
        latency = sorted(r["elapsed_seconds"] for r in group)
        result["variants"][variant] = {"count": len(group), "workflow_passed": sum(r["workflow_passed"] for r in group),
            "necessary_checks_passed": sum(r["checks"]["checks_passed"] for r in group),
            "joint_passed": sum(r["workflow_passed"] and r["checks"]["checks_passed"] for r in group),
            "errors": sum(bool(r["error"]) for r in group), "median_seconds": statistics.median(latency),
            "observed_p95_seconds": latency[max(0, __import__('math').ceil(.95 * len(latency)) - 1)],
            "calls": sum(r["usage"]["current"]["calls"] for r in group if r.get("usage")),
            "input_tokens": sum(r["usage"]["current"]["input_tokens"] for r in group if r.get("usage")),
            "output_tokens": sum(r["usage"]["current"]["output_tokens"] for r in group if r.get("usage")),
            "unknown_usage_records": sum(not r.get("usage") or not r["usage"]["current"]["complete"] for r in group)}
    write_json(Path(output) / "summary.json", result)
    return result


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_path = Path(args.snapshot).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    selected = [c for c in snapshot["cases"] if not args.case or c["id"] in args.case.split(",")]
    if not selected:
        raise ValueError("没有匹配的验收问题")
    if args.case and set(args.case.split(",")) - {c["id"] for c in snapshot["cases"]}:
        raise ValueError("存在未知验收编号，不允许悄悄减少分母")
    variants = args.variants.split(",")
    if set(variants) - {"basic", "research"}:
        raise ValueError("variant 只支持 basic,research")
    if len(variants) != len(set(variants)):
        raise ValueError("variant不能重复，否则会覆盖同题记录")
    source = implementation()
    write_json(output / "manifest.json", {"snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "implementation": source, "cases": [c["id"] for c in selected], "variants": variants,
        "scope": "同源冻结真实模型比较；不包含联网/路由；顺序交替，缓存状态逐任务记录", "seconds_per_case": args.seconds,
        "model_calls_per_case": args.calls, "started_at": time.time(), "configuration": configuration(),
        "settings": "reasoning preserved; analysis cache disabled; isolated DB and vector index"})
    write_json(output / "snapshot.json", snapshot)
    for index, case in enumerate(selected):
        for variant in variants if index % 2 == 0 else list(reversed(variants)):
            if source != implementation():
                raise RuntimeError("验收期间源码发生变化，拒绝混用不同版本；请保留当前结果并新建运行目录")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ANALYSIS_CACHE_ENABLED": "false",
                "RESEARCH_DB": str(output / "research.sqlite3"), "CHECKPOINT_DB": str(output / "checkpoints.sqlite3"),
                "MEMORY_VECTOR_DIR": str(output / "memory"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            path = output / f"{case['id']}-{variant}.json"
            command = [sys.executable, "-X", "utf8", "-m", "agentic_rag.evaluation.acceptance", "worker",
                "--snapshot", str(snapshot_path), "--output", str(path), "--case", case["id"], "--variant", variant,
                "--seconds", str(args.seconds), "--calls", str(args.calls)]
            started = time.perf_counter()
            with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                try:
                    outcome = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.seconds + 45, check=False)
                    failure = f"worker_exit_{outcome.returncode}"
                except subprocess.TimeoutExpired:
                    failure = "worker_wall_timeout"
            if not path.exists():
                write_json(path, {"case_id": case["id"], "variant": variant, "error": {"type": failure},
                    "elapsed_seconds": time.perf_counter()-started, "workflow_passed": False,
                    "checks": {"checks_passed": False}, "usage": None, "manual_review": "not_completed"})
            record = json.loads(path.read_text(encoding="utf-8"))
            if source != implementation():
                raise RuntimeError("当前验收进程运行期间源码发生变化；已保留记录，停止后续实验")
            print(f"{case['id']} {variant}: workflow={record['workflow_passed']} checks={record['checks']['checks_passed']} {record['elapsed_seconds']:.1f}s", flush=True)
            summarize(output, len(selected) * len(variants))
    summary = summarize(output, len(selected) * len(variants))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not summary["missing"] and all(v["joint_passed"] == v["count"] for v in summary["variants"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "worker", "summarize"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--case")
    parser.add_argument("--variant", choices=["basic", "research"])
    parser.add_argument("--variants", default="basic,research")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--calls", type=int, default=40)
    args = parser.parse_args()
    if args.seconds <= 0 or args.calls <= 0:
        parser.error("验收执行预算必须为正数")
    if args.command == "freeze":
        freeze(args.output)
    elif args.command == "run":
        return run(args)
    elif args.command == "worker":
        return worker(args)
    else:
        manifest = json.loads((Path(args.output) / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps(summarize(args.output, len(manifest["cases"]) * len(manifest["variants"])), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
