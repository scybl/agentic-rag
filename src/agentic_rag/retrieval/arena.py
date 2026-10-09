"""检索实验编排：召回、融合、局部重排、去冗余和逐候选轨迹。"""

import time

from .dense import DenseIndex, model_fingerprint
from .fusion import mmr_order, rrf
from .lexical import LexicalIndex
from .reranker import Reranker


class RetrievalArena:
    def __init__(self, documents, config, *, encoder=None, reranker=None):
        self.config = config
        self.documents = {d.document_id: d for d in documents}
        self.lexical = None
        self.dense = None
        self.models = {}
        try:
            if config.adapter in {"bm25", "hybrid", "rerank"}:
                self.lexical = LexicalIndex(documents)
            if config.adapter in {"dense", "hybrid", "rerank"} or config.mmr:
                self.models["embedding"] = model_fingerprint(config.embedding_model) if encoder is None else {"digest": "test-double"}
                self.dense = DenseIndex(documents, config.embedding_model, query_prefix=config.query_prefix, encoder=encoder)
            if config.adapter == "rerank":
                try:
                    self.models["reranker"] = model_fingerprint(config.reranker_model)
                except (OSError, ValueError, TypeError):
                    self.models["reranker"] = {"digest": None, "availability": "unavailable"}
                self.reranker = reranker or Reranker(config.reranker_model, self.models["reranker"]["digest"],
                                                   timeout=config.reranker_timeout)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lexical is not None:
            self.lexical.close()

    def search(self, question):
        config = self.config
        traces, values, timing, warnings = {}, {}, {}, []

        def stage(name, work):
            start = time.perf_counter()
            try:
                output = work()
                for rank, (key, score) in enumerate(output, 1):
                    traces.setdefault(key, []).append({"stage": name, "rank": rank, "score": score})
                    values.setdefault(key, {})[name] = score
                return output
            finally:
                timing[name] = time.perf_counter() - start

        lexical = stage("bm25", lambda: self.lexical.search(question, config.candidate_k)) if self.lexical else []
        dense = stage("dense", lambda: self.dense.search(question, config.candidate_k)) if (
            config.adapter in {"dense", "hybrid", "rerank"}) else []
        if config.adapter == "bm25":
            ranking = lexical
        elif config.adapter == "dense":
            ranking = dense
        else:
            ranking = stage("rrf", lambda: rrf([lexical, dense], constant=config.rrf_constant))
        cache_hit = False
        if config.adapter == "rerank" and ranking:
            subset = ranking[:config.rerank_k]
            def rerank():
                nonlocal cache_hit
                scores, cache_hit = self.reranker.score(question, [
                    self.documents[key].title + " " + self.documents[key].summary for key, _ in subset])
                return sorted(zip([key for key, _ in subset], scores), key=lambda item: (-item[1], item[0]))
            try:
                ranking = stage("rerank", rerank) + ranking[config.rerank_k:]
            except Exception as exc:
                warnings.append({"stage": "rerank", "error_type": type(exc).__name__, "fallback": "rrf"})
        if config.mmr and ranking:
            scores = dict(ranking)
            order = stage("mmr", lambda: [(key, scores[key]) for key in mmr_order(
                ranking, self.dense.by_id, top_k=config.top_k, weight=config.mmr_lambda)])
            ranking = order
        candidates = [{"document_id": key, "rank": rank, "score": score,
                       "components": values[key], "weights": {}, "selected": rank <= config.top_k,
                       "rank_history": traces[key]} for rank, (key, score) in enumerate(ranking, 1)]
        return candidates, {"stage_seconds": timing, "fallbacks": warnings, "rerank_cache_hit": cache_hit}
