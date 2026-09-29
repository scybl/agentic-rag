"""身份、全文版本、覆盖范围、租约和事务投影的回归测试。"""

import time
import pytest
from langchain_core.documents import Document

from agentic_rag.research.store import ResearchStore, split_text


@pytest.fixture
def db(tmp_path):
    return ResearchStore(tmp_path / "research.sqlite3")


def article(text="生猪供给减少。", **metadata):
    return Document(page_content=text, metadata={"article_id": "a1", "title": "行业新闻", "source": "https://example.com/a1",
        "source_type": "news_api", "published_at": "2026-09-29T01:00:00Z", "content_kind": "news_article", **metadata})


def read_result(text):
    return {"claims": [{"kind": "reported_fact", "statement": "原文报道", "quote": text.strip()}], "limitations": []}


def test_chunks_cover_every_character_exactly():
    text = ("第一段，包含日期2026年及价格12.3元。\n" * 40) + "文章结尾"
    chunks = list(split_text(text, 180))
    assert "".join(c[2] for c in chunks) == text
    assert all(text[start:end] == body and end-start <= 180 for start,end,body in chunks)


def test_full_body_change_creates_new_version_with_old_version_retained(db):
    first = db.register(article(), 200)
    same = db.register(article(), 200)
    second = db.register(article("生猪供给增加。"), 200)
    assert same["version"] == first["version"]
    assert second["version"] != first["version"]
    assert db.document(first["version"]).page_content == "生猪供给减少。"


def test_partial_read_is_not_a_complete_memory(db):
    saved = db.register(article("甲" * 450), 200)
    result = db.save_reading(saved, "r1", [(saved["chunks"][0], read_result(saved["chunks"][0]["body"]))])
    assert result["status"] == "partial"
    assert result["covered"] == 1 and result["total"] == 3
    assert db.current_readings("r1") == []


def test_quote_offsets_and_source_are_auditable(db):
    saved = db.register(article("开头介绍。\n价格为12.3元。"), 200)
    result = db.save_reading(saved, "r1", [(saved["chunks"][0], read_result("价格为12.3元。"))])
    claim = result["claims"][0]
    assert saved["body"][claim["start"]:claim["end"]] == claim["quote"]
    assert db.inspect_reading(result["id"])["metadata"]["source"] == "https://example.com/a1"


def test_old_memory_cannot_reactivate_superseded_version(db):
    old = db.register(article(), 200)
    db.save_reading(old, "r1", [(old["chunks"][0], read_result(old["body"]))])
    db.register(article("更新事实。"), 200)
    db.register(article(memory_origin=True), 200)
    assert not db.current_readings("r1")


def test_lease_fences_late_workers(db):
    db.submit("r", "k", "reader", {}, [], 20)
    assert db.claim("k", "old", 1, 3)
    with db.connect() as conn:
        conn.execute("UPDATE tasks SET lease_until=0 WHERE key='k'")
    assert db.claim("k", "new", 30, 3)
    with pytest.raises(RuntimeError, match="租约"):
        db.complete("k", "old", {"bad": True})
    db.complete("k", "new", {"good": True})
    assert db.task("k")["result"] == {"good": True}


def test_task_budget_is_persistent_and_duplicate_submission_is_free(db):
    assert db.submit("r", "a", "reader", {}, [], 1)
    assert db.submit("r", "a", "reader", {}, [], 1)
    assert not db.submit("r", "b", "reader", {}, [], 1)


def test_projection_failure_retains_results_and_is_retryable(db):
    material = db.register(article(), 200)
    result = db.save_reading(material, "r", [(material["chunks"][0], read_result(material["body"]))])
    items = db.pending_vectors("embed-v1")
    for item in items:
        db.vector_synced(item, "embed-v1", "temporary")
    assert db.reading(material["version"], "r")["status"] == "complete"
    assert len(db.pending_vectors("embed-v1")) == len(items)
    for item in items:
        db.vector_synced(item, "embed-v1")
    assert db.pending_vectors("embed-v1") == []
    assert len(db.pending_vectors("embed-v2")) == len(items)


def test_run_lease_rejects_second_process(db):
    db.start_run("r", "问题")
    db.acquire_run("r", "one", 30)
    with pytest.raises(RuntimeError, match="另一个进程"):
        db.acquire_run("r", "two", 30)
    db.release_run("r", "one")
    db.acquire_run("r", "two", 30)


def test_completed_reading_cannot_be_downgraded_by_late_partial_result(db):
    material = db.register(article("甲" * 350), 200)
    results = [(c, read_result(c["body"])) for c in material["chunks"]]
    full = db.save_reading(material, "v1", results)
    late = db.save_reading(material, "v1", results[:1])
    assert late["status"] == "complete" and late["covered"] == full["covered"]


def test_fts_recall_is_versioned_and_accepts_literal_query(db):
    material = db.register(article("能繁母猪存栏减少。"), 200)
    full = db.save_reading(material, "v1", [(material["chunks"][0], read_result(material["body"]))])
    assert db.lexical_readings('能繁母猪 " OR *', "v1") == [full["id"]]
    assert db.lexical_readings("能繁母猪", "v2") == []
    db.register(article("更正后的文章。"), 200)
    assert db.lexical_readings("能繁母猪", "v1") == []
