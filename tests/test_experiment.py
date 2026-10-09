"""离线实验的输入校验、产物完整性、重复性与失败分母。"""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.evaluation.__main__ import main
from agentic_rag.evaluation import runner
from agentic_rag.evaluation.compare import compare_runs, load_run
from agentic_rag.evaluation.contracts import ExperimentConfig, FinanceSuite, digest, load_suite
from agentic_rag.telemetry.exporters import write_json


SUITE = Path(__file__).resolve().parents[1] / "evaluation/suites/finance_smoke_v1/suite.json"


@pytest.fixture
def small_suite(tmp_path):
    data = load_suite(SUITE).model_dump(mode="json")
    data["cases"] = [data["cases"][0], data["cases"][6]]
    path = tmp_path / "suite.json"
    write_json(path, data)
    return path


def test_sample_suite_covers_all_categories_and_no_answer_not_given_fake_positive():
    suite = load_suite(SUITE)
    assert suite.synthetic and len(suite.cases) == 10
    assert len({c.category for c in suite.cases}) == 10
    assert not next(c for c in suite.cases if c.category == "unanswerable").qrels


@pytest.mark.parametrize("change", [
    lambda data: data["cases"].append(copy.deepcopy(data["cases"][0])),
    lambda data: data["documents"].append(copy.deepcopy(data["documents"][0])),
    lambda data: data["cases"][0]["qrels"][0].update(document_id="missing"),
    lambda data: data["cases"][0]["qrels"][0].update(relevance=True),
    lambda data: data["cases"][0]["qrels"].append(copy.deepcopy(data["cases"][0]["qrels"][0])),
    lambda data: data["cases"][0].update(qrels=[]),
    lambda data: data["cases"][0].update(question="   "),
    lambda data: data["documents"][0].update(published_at="2025-03-01"),
    lambda data: data.update(schema_version=999),
])
def test_bad_suite_rejected(change):
    data = load_suite(SUITE).model_dump(mode="json")
    change(data)
    with pytest.raises(ValidationError):
        FinanceSuite.model_validate(data)


def test_repeats_use_existing_ranker_and_artifacts_are_comparable(small_suite, tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    config = ExperimentConfig(repeat=2, split="all")
    runner.run_experiment(small_suite, left, config)
    runner.run_experiment(small_suite, right, config)
    result = compare_runs(left, right)
    assert result["paired_count"] == 4 and result["changed_rankings"] == 0
    assert result["left_failed"] == result["right_failed"] == 0
    manifest, rows = load_run(left)
    assert manifest["scope"].startswith("candidate-ranking-only")
    assert manifest["model_revision"] is None and manifest["prompt_version"] is None
    assert len(rows[0].ranking) == 11 and sum(r.selected for r in rows[0].ranking) == 4
    assert rows[0].ranking[0].document_id == "apple-supply"
    assert rows[0].ranking == rows[2].ranking
    assert (left / "suite.json").exists()


def test_failed_case_kept_and_cli_returns_failure(small_suite, tmp_path, monkeypatch):
    original = runner.current_baseline
    def fail(suite, question, config):
        if "净利润" in question:
            raise TimeoutError("PRIVATE")
        return original(suite, question, config)
    monkeypatch.setattr(runner, "current_baseline", fail)
    output = tmp_path / "out"
    assert main(["run", str(small_suite), "--output", str(output), "--split", "all", "--repeat", "1"]) == 1
    manifest, rows = load_run(output)
    assert manifest["status"] == "completed_with_errors" and len(rows) == 2
    assert rows[1].status == "failed" and rows[1].error_type == "TimeoutError"
    assert "PRIVATE" not in (output / "observations.json").read_text()
    result = compare_runs(output, output)
    assert result["left_failed"] == 1 and result["pairs"][1]["same_ranking"] is None


def test_interruption_preserves_finished_samples_and_cannot_be_compared(small_suite, tmp_path, monkeypatch):
    original = runner.current_baseline
    def interrupt(suite, question, config):
        if "净利润" in question:
            raise KeyboardInterrupt()
        return original(suite, question, config)
    monkeypatch.setattr(runner, "current_baseline", interrupt)
    output = tmp_path / "out"
    assert main(["run", str(small_suite), "--output", str(output), "--split", "all", "--repeat", "1"]) == 130
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "interrupted"
    assert len(json.loads((output / "observations.json").read_text(encoding="utf-8"))) == 1
    with pytest.raises(ValueError, match="未完成"):
        load_run(output)


def test_preflight_does_not_modify_existing_results_or_create_output_for_bad_input(small_suite, tmp_path):
    output = tmp_path / "out"
    runner.run_experiment(small_suite, output, ExperimentConfig(repeat=1))
    before = (output / "manifest.json").read_bytes()
    assert main(["run", str(small_suite), "--output", str(output)]) == 2
    assert before == (output / "manifest.json").read_bytes()
    invalid = tmp_path / "invalid"
    assert main(["run", str(small_suite), "--output", str(invalid), "--repeat", "0"]) == 2
    assert not invalid.exists()


@pytest.mark.parametrize("target", ["suite.json", "observations.json", "trace"])
def test_modified_artifact_rejected(small_suite, tmp_path, target):
    output = tmp_path / "out"
    runner.run_experiment(small_suite, output, ExperimentConfig(repeat=1))
    path = next((output / "traces").glob("*.json")) if target == "trace" else output / target
    data = json.loads(path.read_text(encoding="utf-8"))
    if target == "suite.json":
        data["documents"][0]["summary"] = "Changed"
    elif target == "observations.json":
        data[0]["ranking"][0]["score"] = 999
    else:
        data["spans"][0]["attributes"]["input_tokens"] = 999
    write_json(path, data)
    with pytest.raises(ValueError):
        load_run(output)


def test_missing_observation_rejected_even_with_updated_checksum(small_suite, tmp_path):
    output = tmp_path / "out"
    runner.run_experiment(small_suite, output, ExperimentConfig(repeat=2, split="all"))
    rows = json.loads((output / "observations.json").read_text(encoding="utf-8"))[1:]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(observation_count=len(rows), observations_hash=digest(rows))
    write_json(output / "observations.json", rows)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="样本缺失"):
        load_run(output)


def test_config_change_requires_explicit_variable(small_suite, tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    runner.run_experiment(small_suite, left, ExperimentConfig(repeat=1, top_k=2))
    runner.run_experiment(small_suite, right, ExperimentConfig(repeat=1, top_k=4))
    with pytest.raises(ValueError, match="top_k"):
        compare_runs(left, right)
    comparison = compare_runs(left, right, allow_changes=["top_k"])
    assert comparison["changes"] == {"top_k": [2, 4]}
    assert comparison["changed_rankings"] == 0 and comparison["pairs"][0]["same_selection"] is False


def test_export_cli_does_not_silently_overwrite_file(tmp_path):
    output = tmp_path / "trace.json"
    output.write_text("KEEP", encoding="utf-8")
    assert main(["export-trace", "--database", str(tmp_path / "missing"), "--run-id", "r", "--output", str(output)]) == 2
    assert output.read_text() == "KEEP"


def test_initial_artifact_failure_closes_retrieval_resources(small_suite, tmp_path, monkeypatch):
    from agentic_rag.retrieval import arena
    closed = []
    class FakeArena:
        models = {}
        dense = None
        def __init__(self, *args):
            pass
        def close(self):
            closed.append(True)
    def disk_full(*args):
        raise OSError("disk full")
    monkeypatch.setattr(arena, "RetrievalArena", FakeArena)
    monkeypatch.setattr(runner, "write_json", disk_full)
    with pytest.raises(OSError):
        runner.run_experiment(small_suite, tmp_path / "out", ExperimentConfig(adapter="bm25"))
    assert closed == [True]


def test_retrieval_metadata_must_match_trace(small_suite, tmp_path):
    output = tmp_path / "out"
    runner.run_experiment(small_suite, output, ExperimentConfig(repeat=1))
    rows = json.loads((output / "observations.json").read_text(encoding="utf-8"))
    rows[0]["retrieval"] = {"fallbacks": [{"stage": "rerank"}]}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["observations_hash"] = digest(rows)
    write_json(output / "observations.json", rows)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="检索阶段"):
        load_run(output)
