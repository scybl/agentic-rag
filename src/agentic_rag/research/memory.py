"""可重建的 Chroma 投影：原文、阅读笔记、专题成果及混合召回。"""

import json

from langchain_core.documents import Document

from ..config import settings
from ..news_index import get_embedding_model
from ..news_plan import news_metadata_matches
from .store import fingerprint


class MemoryIndex:
    def __init__(self, store, *, client=None, embeddings=None, embedding_name=None):
        self.store = store
        self._client, self._embeddings = client, embeddings
        self.embedding_name = embedding_name or settings.embedding_model
        self.namespace = fingerprint(self.embedding_name)[:12]

    def collection(self, name):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=settings.memory_vector_dir)
        return self._client.get_or_create_collection(f"{name}_{self.namespace}")

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embedding_model()
        return self._embeddings

    def flush(self, limit=100):
        pending = self.store.pending_vectors(self.embedding_name, limit)
        synced = failed = 0
        for kind in sorted({item["collection"] for item in pending}):
            items = [item for item in pending if item["collection"] == kind]
            try:
                vectors = self.embeddings.embed_documents([item["body"] for item in items])
                self.collection(kind).upsert(ids=[item["id"] for item in items],
                    embeddings=vectors, documents=[item["body"] for item in items],
                    metadatas=[json.loads(item["metadata"]) for item in items])
                for item in items:
                    self.store.vector_synced(item, self.embedding_name)
                synced += len(items)
            except Exception as exc:
                for item in items:
                    self.store.vector_synced(item, self.embedding_name, str(exc)[:500])
                failed += len(items)
        return {"synced": synced, "failed": failed, "remaining": len(self.store.pending_vectors(self.embedding_name, 10000))}

    def recall(self, query, recipe, *, start="", end="", published_after="",
               published_before="", section="", source_names=(), limit=3, subject_terms=()):
        lexical = self.store.lexical_readings(query, recipe)
        semantic_ids = []
        semantic = False
        error = ""
        try:
            collection = self.collection("reading_memory")
            if collection.count():
                response = collection.query(query_embeddings=[self.embeddings.embed_query(query)],
                    n_results=min(50, collection.count()), where={"recipe": recipe}, include=["metadatas"])
                semantic_ids = list(dict.fromkeys(meta["reading_id"] for meta in response["metadatas"][0]))
                semantic = True
        except Exception as exc:
            error = str(exc)[:300]
        # 先用索引召回至多100篇候选，再读取正文，不扫描整个新闻库。
        ids = list(dict.fromkeys(lexical + semantic_ids))
        candidates = []
        filtered = 0
        terms = [term.strip().casefold() for term in subject_terms if term.strip()]
        for row in self.store.current_readings(recipe, ids):
            material = (str(row["metadata"].get("title", "")) + "\n" + row["body"]).casefold()
            if terms and not any(term in material for term in terms):
                filtered += 1
                continue
            if not news_metadata_matches(row["metadata"], {
                "start": start, "end": end,
                "published_after": published_after, "published_before": published_before,
                "section": section, "source_names": list(source_names),
            }):
                continue
            # 新闻记忆不能越权替代知识库，未知/摘要材料也不伪装成全文。
            if row["metadata"].get("source_type") == "news_api":
                candidates.append(row)
        by_id = {row["id"]: row for row in candidates}
        scores = {}
        for ranking in (lexical, semantic_ids):
            for rank, key in enumerate(ranking):
                # SQL 当前版本与日期是权威过滤；旧向量不能自动放行。
                if key in by_id:
                    scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        selected = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
        documents = [Document(page_content=by_id[key]["body"], metadata={**by_id[key]["metadata"],
                     "memory_origin": True, "memory_version": by_id[key]["version_id"]}) for key in selected]
        return documents, {"semantic": semantic, "lexical": len(lexical), "count": len(documents),
                           "subject_filtered": filtered, "subject_terms": list(subject_terms), "error": error}
