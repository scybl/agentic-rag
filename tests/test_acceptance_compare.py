"""同源对照不能静默改标签、比较异源材料或丢失失败。"""

import importlib.util
from pathlib import Path

import pytest
from agentic_rag.telemetry.exporters import write_json


spec = importlib.util.spec_from_file_location("acceptance_compare", Path(__file__).resolve().parents[1] / "scripts/acceptance_compare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair(tmp_path):
    case = {"id": "N01", "question": "是否签署", "sources": ["one"], "quotes": ["尚未签署"],
            "category": "fact", "required": ["未"], "forbidden": []}
    dirs = [tmp_path / "before", tmp_path / "after"]
    for folder in dirs:
        folder.mkdir()
        write_json(folder / "snapshot.json", {"documents": {"one": "尚未签署"}, "cases": [case]})
        write_json(folder / "manifest.json", {"cases": ["N01"], "variants": ["research"]})
        write_json(folder / "summary.json", {"recorded": 1})
        write_json(folder / "N01-research.json", {"case_id": "N01", "variant": "research",
            "workflow_passed": True, "checks": {"checks_passed": False}, "elapsed_seconds": 10,
            "answer": "而非已签署[E1]", "evidence_context": "[E1]尚未签署", "usage": None})
    return dirs


def test_label_fix_is_separate_from_original_score(tmp_path):
    old, new = pair(tmp_path)
    snapshot = module.read(new / "snapshot.json")
    snapshot["cases"][0]["required"] = ["未|而非"]
    write_json(new / "snapshot.json", snapshot)
    result = module.compare(old, new)
    assert result["label_changes"] == ["N01"]
    assert not result["records"][0]["before_original_joint_passed"]
    assert result["records"][0]["before_current_labels_joint_passed"]
    assert result["records"][0]["before_usage"] is None
    assert not module.read(old / "N01-research.json")["checks"]["checks_passed"]


def test_changed_source_cannot_be_labeled_paired(tmp_path):
    old, new = pair(tmp_path)
    snapshot = module.read(new / "snapshot.json")
    snapshot["documents"]["one"] = "已签署"
    write_json(new / "snapshot.json", snapshot)
    with pytest.raises(ValueError, match="来源"):
        module.compare(old, new)


def test_missing_record_is_not_removed_from_denominator(tmp_path):
    old, new = pair(tmp_path)
    (new / "N01-research.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.compare(old, new)


def test_external_labels_do_not_overwrite_either_run(tmp_path):
    old, new = pair(tmp_path)
    labels = module.read(new / "snapshot.json")
    labels["cases"][0]["required"] = ["未|而非"]
    label_file = tmp_path / "labels.json"
    write_json(label_file, labels)
    result = module.compare(old, new, label_file)
    row = result["records"][0]
    assert result["label_changes"] == ["N01"]
    assert not row["after_joint_passed"]
    assert row["after_current_labels_joint_passed"]
    assert module.read(new / "snapshot.json")["cases"][0]["required"] == ["未"]


def test_external_labels_cannot_change_the_question(tmp_path):
    old, new = pair(tmp_path)
    labels = module.read(new / "snapshot.json")
    labels["cases"][0]["question"] = "新的更容易的问题"
    label_file = tmp_path / "labels.json"
    write_json(label_file, labels)
    with pytest.raises(ValueError, match="改变了问题或来源"):
        module.compare(old, new, label_file)
