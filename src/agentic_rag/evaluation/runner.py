"""运行冻结检索实验，保存输入快照、版本与完整候选顺序。"""

import hashlib
import platform
import statistics
import subprocess
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..telemetry.exporters import JsonExporter, write_json
from ..telemetry.schema import Event, Span, Trace, identity
from .contracts import ExperimentConfig, FinanceSuite, digest, load_suite


def implementation_version():
    # 包安装或无 .git 的副本仍能精确定位实际参与排序的源文件。
    package = Path(__file__).resolve().parents[1]
    names = ["news_ranking.py", "news_plan.py", "evaluation/contracts.py", "evaluation/runner.py",
             *(p.relative_to(package).as_posix() for p in sorted((package / "retrieval").glob("*.py")))]
    files = {name: hashlib.sha256((package / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for name in names}
    dependencies = {}
    for name in ("scikit-learn", "numpy", "scipy", "pydantic", "torch", "sentence-transformers", "transformers"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"
    revision = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
                                capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0:
            revision = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"files": files, "digest": digest(files), "git_revision": revision,
            "python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies}


def current_baseline(suite: FinanceSuite, question: str, config: ExperimentConfig):
    from ..news_ranking import rank_candidates
    items = [{"article_id": d.document_id, "candidate_id": d.document_id,
              "title": d.title, "summary": d.summary, "published_at": d.published_at or "",
              "retrieval_queries": [question]} for d in suite.documents]
    # 没有伪造 Dense 分数：这是当前系统已经使用的无向量降级路径。
    ranked = rank_candidates(items, question, {}, config.sort_by)
    return [{"document_id": item["candidate_id"], "rank": index,
             "score": item["priority"]["score"], "components": item["priority"]["components"],
             "weights": item["priority"]["weights"], "selected": index <= config.top_k}
            for index, item in enumerate(ranked, 1)]


def run_experiment(suite_path, output, config: ExperimentConfig):
    suite = load_suite(suite_path)
    cases = [case for case in suite.cases if config.split == "all" or case.split == config.split]
    if not cases:
        raise ValueError("所选 split 没有样本")
    implementation = implementation_version()
    output = Path(output)
    if output.exists():
        raise FileExistsError("实验目录已经存在")
    arena = None
    setup_start = time.perf_counter()
    if config.adapter != "current_tfidf":
        from ..retrieval.arena import RetrievalArena
        arena = RetrievalArena(suite.documents, config)
    # 拒绝复用目录，防止基线被覆盖或不同配置的结果混在一起。
    try:
        output.mkdir(parents=True, exist_ok=False)
    except BaseException:
        if arena is not None:
            arena.close()
        raise
    experiment_id = uuid.uuid4().hex
    snapshot = suite.model_dump(mode="json")
    manifest = {"schema_version": 1, "experiment_id": experiment_id,
                "suite_id": suite.suite_id, "suite_version": suite.version, "suite_hash": suite.fingerprint,
                "corpus_hash": digest(snapshot["documents"]), "synthetic": suite.synthetic,
                "config": config.model_dump(mode="json"), "implementation": implementation,
                "model_revision": arena.models if arena else None, "prompt_version": None,
                "setup_seconds": time.perf_counter() - setup_start,
                "index_vector_bytes": int(arena.dense.vectors.nbytes) if arena and arena.dense else 0,
                "scope": "candidate-ranking-only; no generation or model judge",
                "started_at": time.time(), "status": "running", "observation_count": 0}
    observations = []
    try:
        write_json(output / "suite.json", snapshot)
        write_json(output / "observations.json", observations)
        write_json(output / "manifest.json", manifest)
    except BaseException:
        if arena is not None:
            arena.close()
        raise
    try:
        for repeat in range(1, config.repeat + 1):
            for case in cases:
                started_at, clock = time.time(), time.perf_counter()
                ranking, error, details = [], None, {}
                try:
                    if arena is None:
                        ranking = current_baseline(suite, case.question, config)
                    else:
                        ranking, details = arena.search(case.question)
                except Exception as exc:
                    # 失败独立留档，不能从分母中悄悄消失；错误正文可能含路径或凭据。
                    error = type(exc).__name__
                elapsed = time.perf_counter() - clock
                run_id = identity(experiment_id, case.case_id, str(repeat))
                root, retriever = identity(run_id, "run"), identity(run_id, "retriever")
                status = "failed" if error else "completed"
                attrs = {"case_id": case.case_id, "repeat": repeat, "adapter": config.adapter,
                         "generation_model_calls": 0, "generation_input_tokens": 0, "generation_output_tokens": 0,
                         "embedding_tokens": None, "reranker_tokens": None,
                         "generation_evaluated": False, "error_type": error, "retrieval": details}
                trace = Trace(run_id=run_id, spans=[
                    Span(span_id=root, kind="run", name="ranking experiment", status=status,
                         started_at=started_at, finished_at=time.time(), elapsed_seconds=elapsed, attributes=attrs),
                    Span(span_id=retriever, parent_id=root, kind="retriever", name=config.adapter,
                         status=status, started_at=started_at, finished_at=time.time(), elapsed_seconds=elapsed)],
                    events=[Event(event_id=identity(run_id, "ranking"), span_id=retriever, name="ranking",
                                  at=time.time(), attributes={"candidates": ranking})])
                JsonExporter(output / "traces" / f"{run_id}.json").export(trace)
                observations.append({"case_id": case.case_id, "category": case.category, "split": case.split,
                                     "repeat": repeat, "status": status, "error_type": error,
                                     "elapsed_seconds": elapsed, "ranking": ranking, "trace_id": run_id,
                                     "retrieval": details,
                                     "trace_hash": digest(trace.model_dump(mode="json"))})
                write_json(output / "observations.json", observations)
                manifest["observation_count"] = len(observations)
                write_json(output / "manifest.json", manifest)
        manifest["status"] = "completed_with_errors" if any(o["status"] == "failed" for o in observations) else "completed"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        if arena is not None:
            arena.close()
        manifest.update(finished_at=time.time(), observation_count=len(observations),
                        degraded_count=sum(bool(o["retrieval"].get("fallbacks")) for o in observations),
                        observations_hash=digest(observations))
        write_json(output / "manifest.json", manifest)
    summary = {"observations": len(observations), "failed": sum(o["status"] == "failed" for o in observations),
               "degraded": sum(bool(o["retrieval"].get("fallbacks")) for o in observations),
               "latency_median_seconds": statistics.median(o["elapsed_seconds"] for o in observations),
               "scope": manifest["scope"], "quality_metrics": None}
    write_json(output / "summary.json", summary)
    return manifest
