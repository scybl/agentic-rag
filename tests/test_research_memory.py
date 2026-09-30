"""阅读成果复用、增量分段、真实 Chroma 投影及版本过滤。"""

from unittest.mock import patch
from dataclasses import replace

import pytest
from langchain_core.documents import Document

from agentic_rag.config import settings
from agentic_rag.research import service
from agentic_rag.research.chains import numbered_sentences, resolve_selection, validate_reading
from agentic_rag.research.memory import MemoryIndex
from agentic_rag.research.store import ResearchStore


def document(text, identity="a1", published="2026-09-29T01:00:00Z"):
    return Document(page_content=text, metadata={"article_id": identity, "title": "生猪新闻",
        "source_type": "news_api", "source": "https://example.com/"+identity, "published_at": published, "content_kind": "news_article"})


class NoIndex:
    def flush(self):
        return {"synced": 0, "failed": 0, "remaining": 0}


def reading(payload):
    return {"claims": [{"kind": "reported_fact", "statement": "来源报道", "quote": payload["text"].strip()}], "limitations": []}


def test_second_question_does_not_reread_and_notes_are_not_treated_as_news(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    calls = []
    read = lambda p: calls.append(1) or reading(p)
    state = {"run_id": "r1", "reading_recipe": "recipe1", "question": "价格？", "documents": [document("生猪价格上涨。") ]}
    first = service.read_documents(state, database=db, read_fn=read, index=NoIndex())
    second = service.read_documents({**state, **first, "run_id": "r2", "question": "成本？"}, database=db, read_fn=read, index=NoIndex())
    assert len(calls) == 1
    assert first["reading_reports"][0]["version"] == second["reading_reports"][0]["version"]
    assert first["documents"][0].page_content == second["documents"][0].page_content


def test_only_changed_chunk_is_analyzed_again(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    calls = []
    read = lambda p: calls.append(p["text"]) or reading(p)
    text = "甲" * 99 + "\n" + "乙" * 99 + "\n" + "丙" * 99
    state = {"run_id": "r1", "reading_recipe": "recipe1", "question": "问题", "documents": [document(text)]}
    with patch.object(service, "settings", replace(settings, reading_chunk_chars=100)):
        first = service.read_documents(state, database=db, read_fn=read, index=NoIndex())
        changed = service.read_documents({**state, "run_id": "r2", "documents": [document(text.replace("丙", "丁"))]}, database=db, read_fn=read, index=NoIndex())
    assert len(calls) == 4
    assert first["reading_reports"][0]["version"] != changed["reading_reports"][0]["version"]
    assert changed["reading_reports"][0]["status"] == "complete"


def test_recipe_change_reanalyzes_but_embedding_change_does_not(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    calls = []
    read = lambda p: calls.append(1) or reading(p)
    state = {"run_id": "r1", "reading_recipe": "v1", "question": "问题", "documents": [document("事实。") ]}
    service.read_documents(state, database=db, read_fn=read, index=NoIndex())
    service.read_documents({**state, "run_id": "r2", "reading_recipe": "v2"}, database=db, read_fn=read, index=NoIndex())
    assert len(calls) == 2
    assert db.pending_vectors("new-embedding")
    service.read_documents({**state, "run_id": "r3", "reading_recipe": "v2"}, database=db, read_fn=read, index=NoIndex())
    assert len(calls) == 2


def test_invalid_quote_and_invented_numbers_cannot_be_saved():
    with pytest.raises(ValueError, match="数字"):
        validate_reading({"claims": [{"kind": "reported_fact", "statement": "价格99元", "quote": "价格12元"}], "limitations": []}, "价格12元")
    with pytest.raises(ValueError, match="逐字"):
        validate_reading({"claims": [{"kind": "reported_fact", "statement": "上涨", "quote": "价格99元"}], "limitations": []}, "价格12元")


def test_sentence_selection_fills_exact_quotes_and_corrects_numeric_rewrite():
    text = "本周价格12.3元。\n机构预计明年价格上涨。"
    sentences, _ = numbered_sentences(text)
    result = resolve_selection({"claims": [{"kind": "reported_fact", "statement": "价格99元", "sentence_ids": ["S1"]},
                                          {"kind": "reported_fact", "statement": "机构预测上涨", "sentence_ids": ["S2"]}], "limitations": []}, text, sentences)
    assert result["claims"][0]["statement"] == "本周价格12.3元。"
    assert result["claims"][1]["kind"] == "attributed_forecast"
    assert all(c["quote"] in text for c in result["claims"])


class FakeEmbedding:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        return [1.0, float("生猪" in text), float(len(text)%7), 0.2]


def test_real_chroma_projection_hybrid_recall_and_current_version_filter(tmp_path):
    import chromadb
    db = ResearchStore(tmp_path / "db.sqlite")
    client = chromadb.PersistentClient(path=str(tmp_path / "vectors"))
    index = MemoryIndex(db, client=client, embeddings=FakeEmbedding(), embedding_name="test-v1")
    state = {"run_id": "r1", "reading_recipe": "recipe", "question": "生猪", "documents": [document("生猪价格上涨。") ]}
    service.read_documents(state, database=db, read_fn=reading, index=index)
    docs, info = index.recall("生猪价格", "recipe", start="2026-09-01", end="2026-09-29")
    assert len(docs) == 1 and info["semantic"]
    assert index.recall("生猪", "recipe", published_after="2026-09-29T09:00:01+08:00")[0] == []
    assert len(index.recall("生猪", "recipe", source_names=["example.com"])[0]) == 1
    assert index.recall("生猪", "recipe", source_names=["Reuters"])[0] == []
    assert index.recall("生猪", "recipe", start="2027-01-01")[0] == []
    db.register(document("生猪新闻已经更正。"), 2200)
    assert index.recall("生猪", "recipe")[0] == []
    assert index.flush()["failed"] == 0


def test_failed_vector_write_retries_without_reader(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    broken = MemoryIndex(db, embeddings=FakeEmbedding(), embedding_name="broken")
    state = {"run_id": "r1", "reading_recipe": "recipe", "documents": [document("生猪事实。") ]}
    with patch.object(broken, "collection", side_effect=RuntimeError("offline")):
        result = service.read_documents(state, database=db, read_fn=reading, index=broken)
    assert result["reading_reports"][0]["status"] == "complete"
    assert db.pending_vectors("broken")
