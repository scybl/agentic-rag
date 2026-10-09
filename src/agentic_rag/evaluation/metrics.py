"""ranx 标准指标；不把重复运行视为独立样本，也不隐去失败查询。"""

import math
import os
import statistics
import tempfile
import uuid
import importlib
from importlib.metadata import version
from pathlib import Path

from .compare import compare_runs, load_run
from .contracts import load_suite
from .contracts import digest


def load_ranx():
    # Numba 默认探测 site-packages 是否可写；Windows 受限安装目录下 tempfile
    # 可能反复尝试。用进程级临时缓存，并通过一次独占创建尽早发现权限问题。
    cache = Path(os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "agentic-rag-numba")))
    cache.mkdir(parents=True, exist_ok=True)
    probe = cache / ("write-check-" + uuid.uuid4().hex)
    with probe.open("x", encoding="utf-8"):
        pass
    probe.unlink()
    # ranx 会导入数据集/绘图库；本项目评分不下载数据集，也不需要用户主目录缓存。
    os.environ.setdefault("IR_DATASETS_HOME", str(Path(tempfile.gettempdir()) / "agentic-rag-ir-datasets"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "agentic-rag-matplotlib"))
    try:
        return importlib.import_module("ranx")
    except ImportError as exc:
        raise ValueError('请先安装检索评测依赖：python -m pip install -e ".[retrieval-eval]"') from exc


def metric_names(cutoffs):
    if not cutoffs or any(type(k) is not int or not 1 <= k <= 1000 for k in cutoffs):
        raise ValueError("指标 cutoff 须为 1–1000 的整数")
    return [f"{name}@{k}" for k in sorted(set(cutoffs)) for name in ("precision", "recall", "mrr", "ndcg")]


def inputs(directory, rows, repeat):
    suite = load_suite(Path(directory) / "suite.json")
    cases = {c.case_id: c for c in suite.cases}
    selected = [row for row in rows if row.repeat == repeat]
    qrels = {row.case_id: {q.document_id: q.relevance for q in cases[row.case_id].qrels}
             for row in selected if any(q.relevance > 0 for q in cases[row.case_id].qrels)}
    # 用严格降序的排名代理分数，把 MMR 后的最终位置传给 ranx，避免它重按原始分数排序。
    run = {row.case_id: {r.document_id: 1.0 / r.rank for r in row.ranking}
           for row in selected if row.case_id in qrels}
    excluded = [row.case_id for row in selected if row.case_id not in qrels]
    return qrels, run, excluded


def make_run(ranx, rankings, qrels, name=None):
    # ranx 0.3 的字典构造器要求至少一个文档，全部失败/零命中时不能直接传空字典。
    # 用公开的空 Run + make_comparable 补齐空结果，不伪造占位文档或删除失败问题。
    nonempty = {key: values for key, values in rankings.items() if values}
    return ranx.Run(nonempty or None, name=name).make_comparable(qrels)


def score_run(directory, cutoffs=(1, 4, 10)):
    ranx = load_ranx()
    manifest, rows = load_run(directory)
    names = metric_names(cutoffs)
    per_case, excluded = {}, []
    for repeat in range(1, manifest["config"]["repeat"] + 1):
        qrels, rankings, excluded = inputs(directory, rows, repeat)
        if not qrels:
            continue
        labels = ranx.Qrels(qrels)
        run = make_run(ranx, rankings, labels)
        ranx.evaluate(labels, run, names, threads=1)
        for name in names:
            for case_id, value in run.scores[name].items():
                per_case.setdefault(case_id, {}).setdefault(name, []).append(float(value))
    scores = {key: {name: statistics.mean(values) for name, values in metrics.items()}
              for key, metrics in per_case.items()}
    categories = {row.case_id: row.category for row in rows}
    by_category = {}
    for category in sorted(set(categories.values())):
        bucket = [scores[key] for key in scores if categories[key] == category]
        by_category[category] = {"scored_questions": len(bucket), "metrics": {
            name: statistics.mean(item[name] for item in bucket) if bucket else None for name in names}}
    return {"schema_version": 1, "experiment_id": manifest["experiment_id"], "suite_hash": manifest["suite_hash"],
            "implementation_digest": manifest["implementation"]["digest"], "ranx_version": version("ranx"),
            "metric_protocol": {"cutoffs": sorted(set(cutoffs)), "aggregation": "repeat-mean then question-macro",
                                "ranking_scores": "reciprocal-final-rank", "relevance_threshold": 1,
                                "source_digest": digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                                                         for name in ("metrics.py", "compare.py")})},
            "synthetic": manifest["synthetic"], "scored_questions": len(scores),
            "excluded_no_positive_qrels": excluded,
            "failed_observations": sum(row.status == "failed" for row in rows),
            "degraded_observations": sum(bool(row.retrieval.get("fallbacks")) for row in rows),
            "metrics": {name: statistics.mean(s[name] for s in scores.values()) if scores else None for name in names},
            "by_category": by_category, "per_question": scores, "per_repeat": per_case,
            "limitations": ["相关性衡量的是召回/排序，不是最终答案正确率；未标注文档按不相关处理。",
                            "无正相关标注的问题不能计算有意义的 Recall/nDCG，单独列出，未从失败统计中删除。",
                            "先在题内平均重复运行，再按题宏平均；重复运行不会扩大独立问题数。"]}


def _json_numbers(value):
    if isinstance(value, dict):
        return {str(k): _json_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numbers(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare_quality(left, right, *, cutoffs=(1, 4, 10), allow_changes=()):
    behavioral = compare_runs(left, right, allow_changes=allow_changes)
    left_scores, right_scores = score_run(left, cutoffs), score_run(right, cutoffs)
    _, a = load_run(left)
    _, b = load_run(right)
    qrels, left_rank, _ = inputs(left, a, 1)
    _, right_rank, _ = inputs(right, b, 1)
    statistics_report, reason = None, None
    # 随机/失败/缓存变化导致各次排名不同时，不拿第一轮冒充各轮的平均质量。
    def stable(rows):
        first = {}
        for row in rows:
            value = (row.status, tuple(r.document_id for r in row.ranking))
            if row.case_id in first and first[row.case_id] != value:
                return False
            first[row.case_id] = value
        return True
    if len(qrels) < 2:
        reason = "至少需要两个有正相关标注的独立问题"
    elif not stable(a) or not stable(b):
        reason = "重复运行排名或状态不稳定；请先分析波动，不能放大为独立样本"
    else:
        ranx = load_ranx()
        labels = ranx.Qrels(qrels)
        report = ranx.compare(labels, [make_run(ranx, left_rank, labels, "left"),
                                      make_run(ranx, right_rank, labels, "right")],
                         metric_names(cutoffs), stat_test="fisher", n_permutations=1000,
                         random_seed=42, threads=1, max_p=0.05)
        statistics_report = _json_numbers(report.to_dict())
    return {"left": left_scores, "right": right_scores, "changes": behavioral["changes"],
            "statistical_test": statistics_report, "test_unavailable_reason": reason,
            "test_protocol": {"test": "fisher", "permutations": 1000, "seed": 42,
                              "independent_unit": "question", "multiple_testing_correction": None},
            "limitations": ["合成小样本的显著性只用于验证实验流程，不支持真实领域收益声明。",
                            "多个指标的 p 值未进行多重比较校正，属于探索性结果，不能挑选有利指标宣称整体显著。"]}
