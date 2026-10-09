"""用真实 ranx 验证口径；可选依赖未安装时跳过，不用自写算法冒充 ranx。"""

import importlib.util
import math

import pytest

from agentic_rag.evaluation import runner
from agentic_rag.evaluation.contracts import ExperimentConfig
from agentic_rag.evaluation.metrics import compare_quality, load_ranx, score_run
from agentic_rag.telemetry.exporters import write_json


pytestmark = pytest.mark.skipif(importlib.util.find_spec("ranx") is None, reason="需要 retrieval-eval 可选依赖")


@pytest.fixture
def suite(tmp_path):
    documents = [{"document_id": key, "title": key, "summary": key, "body": key,
                  "published_at": None, "source": "synthetic"} for key in ("a", "b")]
    cases = [{"case_id": key, "question": key, "category": "unanswerable" if key == "none" else "fact",
              "split": "dev", "answerable": key != "none",
              "qrels": [] if key == "none" else [{"document_id": key, "relevance": 3}],
              "reference_answer": "synthetic", "key_evidence": ["synthetic"], "forbidden_claims": ["invented"],
              "risk_obligations": [], "rubric": ["test"]} for key in ("a", "b", "none")]
    path = tmp_path / "suite.json"
    write_json(path, {"schema_version": 1, "suite_id": "test", "version": "1", "provenance": "test",
                      "synthetic": True, "documents": documents, "cases": cases})
    return path


def ranking(suite, question, config):
    return [{"document_id": key, "rank": rank, "score": 1 / rank, "components": {}, "weights": {},
             "selected": rank <= config.top_k} for rank, key in enumerate(("a", "b"), 1)]


def test_real_ranx_metrics_no_positive_exclusion_and_repeat_denominator(suite, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "current_baseline", ranking)
    output = tmp_path / "out"
    runner.run_experiment(suite, output, ExperimentConfig(repeat=2))
    report = score_run(output, [1, 2])
    assert report["scored_questions"] == 2  # 不是重复两轮后的 4，也不是无答案样本加进来的 3。
    assert report["excluded_no_positive_qrels"] == ["none"]
    assert report["metrics"]["precision@1"] == 0.5
    assert report["metrics"]["recall@1"] == 0.5
    assert report["metrics"]["recall@2"] == 1
    assert report["metrics"]["mrr@2"] == 0.75
    assert report["metrics"]["ndcg@2"] == pytest.approx((1 + 1 / math.log2(3)) / 2)


def test_failed_eligible_query_is_not_dropped_from_metric_mean(suite, tmp_path, monkeypatch):
    def fail(suite, question, config):
        if question == "a":
            raise TimeoutError()
        return ranking(suite, question, config)
    monkeypatch.setattr(runner, "current_baseline", fail)
    output = tmp_path / "out"
    runner.run_experiment(suite, output, ExperimentConfig(repeat=1))
    report = score_run(output, [2])
    assert report["scored_questions"] == 2 and report["failed_observations"] == 1
    assert report["metrics"]["recall@2"] == 0.5
    assert report["per_question"]["a"]["recall@2"] == 0


def test_real_paired_test_and_unstable_repeats_are_not_pseudoreplicated(suite, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "current_baseline", ranking)
    left, right = tmp_path / "a", tmp_path / "b"
    runner.run_experiment(suite, left, ExperimentConfig(repeat=2))
    runner.run_experiment(suite, right, ExperimentConfig(repeat=2))
    result = compare_quality(left, right, cutoffs=[1])
    assert result["statistical_test"]["stat_test"] == "fisher"
    assert result["test_protocol"]["independent_unit"] == "question"
    count = 0
    def changing(suite, question, config):
        nonlocal count
        count += 1
        if count == 1:
            raise TimeoutError()
        return ranking(suite, question, config)
    monkeypatch.setattr(runner, "current_baseline", changing)
    unstable = tmp_path / "unstable"
    runner.run_experiment(suite, unstable, ExperimentConfig(repeat=2))
    result = compare_quality(left, unstable, cutoffs=[1])
    assert result["statistical_test"] is None and "不稳定" in result["test_unavailable_reason"]


def test_cache_path_failure_is_immediate_before_library_import(tmp_path, monkeypatch):
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(blocked))
    with pytest.raises(FileExistsError):
        load_ranx()


def test_all_failed_queries_still_score_zero(suite, tmp_path, monkeypatch):
    def fail(*args):
        raise TimeoutError()
    monkeypatch.setattr(runner, "current_baseline", fail)
    output = tmp_path / "failed"
    runner.run_experiment(suite, output, ExperimentConfig(repeat=1))
    report = score_run(output, [1])
    assert report["failed_observations"] == 3
    assert report["scored_questions"] == 2
    assert all(value == 0 for value in report["metrics"].values())
    compared = compare_quality(output, output, cutoffs=[1])
    assert compared["statistical_test"]["stat_test"] == "fisher"


def test_no_positive_qrels_returns_unknown_not_perfect_score(suite, tmp_path):
    import json
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["cases"] = [data["cases"][-1]]
    write_json(suite, data)
    output = tmp_path / "no-positive"
    runner.run_experiment(suite, output, ExperimentConfig(repeat=1))
    report = score_run(output, [1])
    assert report["scored_questions"] == 0
    assert report["excluded_no_positive_qrels"] == ["none"]
    assert all(value is None for value in report["metrics"].values())
