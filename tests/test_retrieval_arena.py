"""检索组件与进程边界，涵盖空召回、负分、重复候选、超时和真实 FTS5。"""

import multiprocessing
from pathlib import Path
import time

import numpy as np
import pytest
from pydantic import ValidationError

from agentic_rag.evaluation.contracts import ExperimentConfig, FrozenDocument
from agentic_rag.retrieval.arena import RetrievalArena
from agentic_rag.retrieval.dense import DenseIndex, model_fingerprint, normalize
from agentic_rag.retrieval.fusion import mmr_order, rrf
from agentic_rag.retrieval.lexical import LexicalIndex, tokens
from agentic_rag.retrieval.reranker import Reranker, bounded_predict, local_predict


def document(key, title):
    return FrozenDocument(document_id=key, title=title, summary=title, body=title,
                          source="synthetic", published_at=None)


DOCS = [document("a", "苹果库存下降"), document("b", "苹果手机发布"), document("c", "原油运输成本")]


class Encoder:
    def encode(self, texts, **kwargs):
        return [[1, 0], [0.95, 0.05], [0, 1]] if len(texts) == 3 else [[1, 0]]


def delayed_write(path):
    time.sleep(3)
    Path(path).write_text("should not happen", encoding="utf-8")
    return [1.0]


def scores_worker():
    return [0.1, 0.2]


def failed_worker():
    raise ValueError("PRIVATE")


def test_fts5_chinese_frequency_empty_and_literal_query():
    assert tokens("苹果苹果") == ["苹果", "果苹", "苹果"]
    index = LexicalIndex(DOCS)
    try:
        assert index.search("库存", 3)[0][0] == "a"
        assert index.search("不存在xyz", 3) == []
        assert index.search("***", 3) == []
        # 用户输入的引号、字段、运算符均不能执行成 FTS 查询语法。
        assert {key for key, _ in index.search('" OR title:* 苹果', 10)} <= {"a", "b", "c"}
        assert all(value > 0 for _, value in index.search("苹果", 3))
    finally:
        index.close()


def test_dense_normalizes_negative_scores_and_preserves_stable_ties():
    index = DenseIndex(DOCS, encoder=Encoder())
    assert [key for key, _ in index.search("苹果", 3)] == ["a", "b", "c"]
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1)
    with pytest.raises(ValueError, match="零向量"):
        normalize([[0, 0]])
    with pytest.raises(ValueError, match="有限"):
        normalize([[float("nan"), 1]])


def test_rrf_uses_rank_not_incomparable_raw_score_and_rejects_duplicates():
    rankings = [[("a", -500), ("b", -1)], [("b", 0.2), ("a", 999)]]
    result = rrf(rankings)
    assert result[0][0] == "a"  # 相同融合分按 ID 稳定排序。
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert result == rrf([[('a', 1), ('b', 2)], [('b', 100), ('a', 0)]])
    assert rrf([[], []]) == []
    with pytest.raises(ValueError, match="重复"):
        rrf([[('a', 1), ('a', 2)]])


def test_mmr_can_choose_diverse_document_and_weight_one_retains_order():
    ranking = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    vectors = {"a": np.array([1., 0.]), "b": np.array([1., 0.]), "c": np.array([0., 1.])}
    assert mmr_order(ranking, vectors, top_k=2, weight=0.2) == ["a", "c", "b"]
    assert mmr_order(ranking, vectors, top_k=2, weight=1) == ["a", "b", "c"]


def test_arena_tracks_every_stage_and_does_not_invent_unmatched_lexical_candidates():
    config = ExperimentConfig(adapter="hybrid", embedding_model="test-double", candidate_k=2, top_k=2, mmr=True)
    arena = RetrievalArena(DOCS, config, encoder=Encoder())
    try:
        ranking, details = arena.search("苹果库存")
        assert len(ranking) == 2 and [row["rank"] for row in ranking] == [1, 2]
        assert all({h["stage"] for h in row["rank_history"]} >= {"dense", "rrf", "mmr"} for row in ranking)
        assert not details["fallbacks"]
        assert details["stage_seconds"].keys() == {"bm25", "dense", "rrf", "mmr"}
    finally:
        arena.close()
    lexical = RetrievalArena(DOCS, ExperimentConfig(adapter="bm25"))
    try:
        ranking, _ = lexical.search("不存在xyz")
        assert ranking == []
    finally:
        lexical.close()


def test_rerank_only_changes_top_n_and_success_cache_is_explicit():
    calls = []
    reranker = Reranker(None, "test", predict=lambda question, texts: calls.append(texts) or [-5.0, 5.0])
    config = ExperimentConfig(adapter="rerank", embedding_model="test", reranker_model="test", rerank_k=2)
    arena = RetrievalArena(DOCS, config, encoder=Encoder(), reranker=reranker)
    try:
        first, a = arena.search("苹果库存")
        second, b = arena.search("苹果库存")
        assert first == second and len(calls) == 1
        assert not a["rerank_cache_hit"] and b["rerank_cache_hit"]
        assert first[0]["document_id"] == "b" and first[-1]["document_id"] == "c"
        assert "rerank" not in first[-1]["components"]
        assert all(len(row["rank_history"]) >= 2 for row in first)
    finally:
        arena.close()


@pytest.mark.parametrize("failure", [TimeoutError, FileNotFoundError, ValueError])
def test_rerank_failures_preserve_fused_order_and_are_not_cached(failure):
    attempts = []
    def broken(*args):
        attempts.append(1)
        raise failure("PRIVATE")
    config = ExperimentConfig(adapter="rerank", embedding_model="test", reranker_model="test")
    arena = RetrievalArena(DOCS, config, encoder=Encoder(), reranker=Reranker(None, "test", predict=broken))
    try:
        a, metadata = arena.search("苹果库存")
        b, _ = arena.search("苹果库存")
        assert [row["document_id"] for row in a] == [row["document_id"] for row in b]
        assert len(attempts) == 2 and metadata["fallbacks"] == [
            {"stage": "rerank", "error_type": failure.__name__, "fallback": "rrf"}]
        assert "PRIVATE" not in str(metadata)
    finally:
        arena.close()


def test_subprocess_timeout_stops_worker_and_does_not_leave_processes(tmp_path):
    before = {p.pid for p in multiprocessing.active_children()}
    output = tmp_path / "late.txt"
    with pytest.raises(TimeoutError):
        bounded_predict(delayed_write, (str(output),), 0.1)
    assert not output.exists()
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_subprocess_returns_scores_and_masks_error_text():
    assert bounded_predict(scores_worker, (), 20) == [0.1, 0.2]
    with pytest.raises(RuntimeError, match="ValueError") as error:
        bounded_predict(failed_worker, (), 20)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("values", [[float("nan")], [[0.1, 0.2]], [True], []])
def test_malformed_rerank_scores_rejected(values):
    reranker = Reranker(None, "test", predict=lambda *args: values)
    with pytest.raises(ValueError):
        reranker.score("q", ["doc"])
    assert not reranker.cache


def test_model_fingerprint_changes_with_weights_and_ignores_git(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"v1")
    before = model_fingerprint(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.json").write_text("IGNORE", encoding="utf-8")
    assert model_fingerprint(tmp_path) == before
    weights.write_bytes(b"v2")
    assert model_fingerprint(tmp_path)["digest"] != before["digest"]


@pytest.mark.parametrize("args", [dict(adapter="dense"), dict(adapter="rerank", embedding_model="x"),
                                  dict(adapter="bm25", sort_by="newest"), dict(candidate_k=0)])
def test_invalid_arena_configuration_rejected(args):
    with pytest.raises(ValidationError):
        ExperimentConfig(**args)


def test_embedding_model_cannot_masquerade_as_trained_reranker(tmp_path):
    (tmp_path / "config.json").write_text('{"architectures": ["BertModel"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="随机分类头"):
        local_predict(str(tmp_path), [["q", "text"]])
