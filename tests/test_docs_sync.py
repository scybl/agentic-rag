"""文档门禁自身的回归：能发现漂移，而不是仅测试当前快照恰好相等。"""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_sync", ROOT / "scripts/docs_sync.py")
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


@pytest.fixture
def replica(tmp_path):
    inventory = docs.inventories(ROOT)
    for name in [*inventory["sources"], *inventory["documents"], docs.LOCK]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path


def change(root, name, before, after):
    path = root / name
    text = docs.read(root, name)
    assert before in text
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


def test_repository_documentation_is_reviewed_and_current():
    assert docs.check() == []


def test_reference_ignores_personal_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "PRIVATE_MODEL_MUST_NOT_LEAK")
    monkeypatch.setenv("NEWS_API_KEY", "SECRET_MUST_NOT_LEAK")
    result = docs.render_reference(ROOT)
    assert "PRIVATE_MODEL_MUST_NOT_LEAK" not in result
    assert "SECRET_MUST_NOT_LEAK" not in result
    assert "qwen2.5:3b" in result


def test_both_graph_modes_are_extracted():
    research = docs.graph_edges(ROOT, True)
    basic = docs.graph_edges(ROOT, False)
    assert ("supplement_sources", "直接", "grade_documents") in research
    assert ("grade_documents", "直接", "read_documents") in research
    assert ("read_documents", "直接", "assess_evidence") in research
    assert ("supplement_sources", "直接", "grade_documents") in basic
    assert ("assess_evidence", "generate", "dispatch_specialists") in research
    assert ("evaluate_generation", "finish", "END") in basic
    assert not any("initialize_research" in row for row in basic)


def test_changed_default_stales_reference_and_review(replica):
    change(replica, "src/agentic_rag/config.py", '"RETRIEVAL_K", "4"', '"RETRIEVAL_K", "7"')
    errors = docs.check(replica)
    assert any("代码事实参考过时" in e for e in errors)
    assert any("[sources]" in e and "config.py" in e for e in errors)


def test_changed_implementation_requires_review_even_with_same_signature(replica):
    path = replica / "src/agentic_rag/tools/execution.py"
    path.write_text(docs.read(replica, path) + "\n# implementation reviewed separately\n", encoding="utf-8")
    assert any("execution.py" in e for e in docs.check(replica))


@pytest.mark.parametrize("kind", ["source", "document"])
def test_new_files_are_detected(replica, kind):
    name = "src/agentic_rag/new_source.py" if kind == "source" else "docs/new-guide.md"
    (replica / name).write_text("# New\n", encoding="utf-8")
    assert any(name in e for e in docs.check(replica))


def test_deleted_document_and_broken_link_are_detected(replica):
    (replica / "docs/tools-guide.md").unlink()
    errors = docs.check(replica)
    assert any("[documents]" in e and "tools-guide.md" in e for e in errors)
    assert any("本地链接不存在" in e for e in errors)


def test_html_symbol_move_can_be_refreshed(replica):
    html_path = replica / "docs/code-links.html"
    line = docs.source_line(replica, {"data-source-path": "src/agentic_rag/tools/news_reading.py", "data-source-symbol": "read_news"})
    html_path.write_text(f'<a href="../src/agentic_rag/tools/news_reading.py" data-source-path="src/agentic_rag/tools/news_reading.py" '
                        f'data-source-symbol="read_news" data-source-line="{line}">代码</a><span class="locator">第 {line} 行</span>', encoding="utf-8")
    path = replica / "src/agentic_rag/tools/news_reading.py"
    path.write_text("\n\n" + docs.read(replica, path), encoding="utf-8")
    assert any("代码定位过时" in e for e in docs.validate_links(replica))
    html = docs.refresh_locations(replica, docs.read(replica, html_path))
    html_path.write_text(html, encoding="utf-8")
    assert not any("代码定位过时" in e for e in docs.validate_links(replica))


def test_refresh_does_not_acknowledge_review(replica, monkeypatch):
    monkeypatch.setattr(docs, "ROOT", replica)
    saved = docs.read(replica, docs.LOCK)
    change(replica, "src/agentic_rag/config.py", '"RETRIEVAL_K", "4"', '"RETRIEVAL_K", "7"')
    assert docs.main(["--refresh"]) == 0
    assert docs.read(replica, docs.LOCK) == saved
    assert any("[sources]" in e for e in docs.check(replica))


@pytest.mark.parametrize("name", ["docs/Agent求职面试题与参考答案.html", "docs/Interview-notes.md", "docs/iNtErViEw/local.html"])
def test_private_material_is_optional_unread_and_untouched(replica, monkeypatch, name):
    assert docs.check(replica) == []  # 干净克隆无需个人资料即可检查。
    path = replica / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\x00private")  # 不是合法 UTF-8：排除必须发生在读取之前。
    assert docs.inventories(replica) == docs.inventories(ROOT)
    monkeypatch.setattr(docs, "ROOT", replica)
    assert docs.main(["--refresh"]) == 0
    assert docs.main(["--acknowledge-review"]) == 0
    assert docs.check(replica) == []
    assert path.read_bytes() == b"\xff\x00private"
    assert name not in docs.read(replica, docs.LOCK)


def test_public_link_to_private_material_is_rejected_even_when_local_file_exists(replica):
    (replica / "docs/面试笔记.html").write_text("private", encoding="utf-8")
    path = replica / "docs/index.md"
    path.write_text(docs.read(replica, path) + "\n[private](面试笔记.html)\n", encoding="utf-8")
    assert any("公开文档不能依赖" in e for e in docs.validate_links(replica))


def test_line_endings_do_not_create_false_drift(replica):
    path = replica / "src/agentic_rag/config.py"
    text = docs.read(replica, path)
    path.write_bytes(text.replace("\n", "\r\n").encode())
    assert docs.inventories(replica) == docs.inventories(ROOT)


def test_broken_html_anchor_is_detected(replica):
    (replica / "docs/code-links.html").write_text('<div id="overview"><a href="#missing-anchor">错误</a></div>', encoding="utf-8")
    assert any("HTML 锚点不存在" in e for e in docs.validate_links(replica))


def test_invalid_svg_is_detected(replica):
    (replica / "docs/agentic-rag-core-architecture.svg").write_text("<svg>", encoding="utf-8")
    assert any("SVG" in e for e in docs.validate_links(replica))


def test_unrecognized_graph_branch_fails_loudly(replica):
    change(replica, "src/agentic_rag/graph/build.py", "if research:", "if another_flag:")
    with pytest.raises(ValueError, match="新的条件"):
        docs.graph_edges(replica, True)


def test_review_confirmation_refuses_stale_generated_facts(replica, monkeypatch):
    change(replica, "src/agentic_rag/config.py", '"RETRIEVAL_K", "4"', '"RETRIEVAL_K", "7"')
    monkeypatch.setattr(docs, "ROOT", replica)
    assert docs.main(["--acknowledge-review"]) == 1
