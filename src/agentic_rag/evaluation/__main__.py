"""离线检索评测和只读取证；不加载个人配置，按需加载本地检索模型。"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from ..telemetry.adapters import research_trace
from ..telemetry.exporters import JsonExporter, read_research, write_json
from .compare import compare_runs
from .contracts import ExperimentConfig, load_suite
from .runner import run_experiment


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="冻结财经样本实验与运行 Trace 导出")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="校验材料与相关性标注")
    validate.add_argument("suite", type=Path)
    run = commands.add_parser("run", help="运行冻结输入上的 TF-IDF/BM25/Dense/混合检索实验")
    run.add_argument("suite", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repeat", type=int, default=2)
    run.add_argument("--top-k", type=int, default=4)
    run.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    run.add_argument("--sort-by", choices=["relevance", "newest", "oldest"], default="relevance")
    run.add_argument("--adapter", choices=["current_tfidf", "bm25", "dense", "hybrid", "rerank"], default="current_tfidf")
    run.add_argument("--candidate-k", type=int, default=100)
    run.add_argument("--rrf-constant", type=int, default=60)
    run.add_argument("--rerank-k", type=int, default=80)
    run.add_argument("--embedding-model")
    run.add_argument("--query-prefix", default="为这个句子生成表示以用于检索相关文章：")
    run.add_argument("--reranker-model")
    run.add_argument("--reranker-timeout", type=float, default=60)
    run.add_argument("--mmr", action="store_true")
    run.add_argument("--mmr-lambda", type=float, default=0.7)
    compare = commands.add_parser("compare", help="先验证快照与产物，再配对比较")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--allow-change", choices=sorted(set(ExperimentConfig.model_fields) - {"repeat", "split"}) + ["implementation"], action="append", default=[])
    compare.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-trace", help="只读导出 SQLite 中的完整运行记录")
    export.add_argument("--database", type=Path, required=True)
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="使用 ranx 计算标准检索指标及可选配对检验")
    score.add_argument("directory", type=Path)
    score.add_argument("--against", type=Path)
    score.add_argument("--k", type=int, action="append")
    score.add_argument("--allow-change", choices=sorted(set(ExperimentConfig.model_fields) - {"repeat", "split"}) + ["implementation"], action="append", default=[])
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            suite = load_suite(args.suite)
            print(json.dumps({"suite_id": suite.suite_id, "suite_hash": suite.fingerprint,
                              "documents": len(suite.documents), "cases": len(suite.cases),
                              "synthetic": suite.synthetic}, ensure_ascii=False, indent=2))
        elif args.command == "run":
            config = ExperimentConfig(**{key: getattr(args, key) for key in ExperimentConfig.model_fields})
            result = run_experiment(args.suite, args.output, config)
            print(f"实验：{result['experiment_id']}；状态：{result['status']}；降级：{result['degraded_count']}；结果：{args.output}")
            return 1 if result["status"] != "completed" else 0
        elif args.command == "compare":
            if args.output.exists():
                raise FileExistsError("输出已存在，请选择新文件")
            result = compare_runs(args.left, args.right, allow_changes=args.allow_change)
            write_json(args.output, result)
            print(f"配对 {result['paired_count']} 条；排名变化 {result['changed_rankings']}；结果：{args.output}")
        elif args.command == "score":
            if args.output.exists():
                raise FileExistsError("输出已存在，请选择新文件")
            from .metrics import compare_quality, score_run
            result = (compare_quality(args.directory, args.against, cutoffs=args.k or (1, 4, 10),
                                      allow_changes=args.allow_change) if args.against else
                      score_run(args.directory, args.k or (1, 4, 10)))
            write_json(args.output, result)
            print(f"检索质量报告：{args.output}")
        else:
            if args.output.exists():
                raise FileExistsError("输出已存在，请选择新文件")
            run_info, records = read_research(args.database, args.run_id)
            trace = research_trace(run_info, records)
            JsonExporter(args.output).export(trace)
            print(f"导出 {len(trace.spans)} 个 Span、{len(trace.events)} 个事件；结果：{args.output}")
        return 0
    except KeyboardInterrupt:
        print("实验已中断；已完成记录保留，不能作为完整实验比较。", file=sys.stderr)
        return 130
    except ValidationError as exc:
        # 不打印 pydantic 错误里的 input，避免把完整材料输出到终端。
        locations = [".".join(map(str, e["loc"])) or "root" for e in exc.errors()]
        print("协议校验失败：" + ", ".join(locations), file=sys.stderr)
        return 2
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        print(f"执行失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
