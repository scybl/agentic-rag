"""为只读 API 获取的新闻建立增量本机 Chroma 索引。

第一阶段只索引标题和摘要。文章正文仍保留在服务器上，需要时可通过
NewsClient.article(article_id) 获取。
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT, settings
from .news_api import NewsClient

DEFAULT_INDEX_DIR = PROJECT_ROOT / ".chroma" / "news"
DEFAULT_COLLECTION = "tonghuashun_news"


@dataclass(frozen=True)
class IndexRecord:
    article_id: str
    text: str
    metadata: dict[str, str]


@dataclass
class SyncStats:
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    invalid: int = 0


def prepare_record(item: dict[str, Any]) -> IndexRecord | None:
    """将一条 API 数据规范化为紧凑且结果确定的索引记录。"""
    article_id = str(item.get("article_id") or "").strip()
    title = str(item.get("title") or "").strip()
    if not article_id or not title:
        return None
    summary = str(item.get("summary") or "").strip()
    text = f"{title}\n{summary}" if summary else title
    metadata = {
        "article_id": article_id,
        "title": title,
        "published_at": str(item.get("published_at") or ""),
        "section": str(item.get("section") or ""),
        "url": str(item.get("canonical_url") or item.get("url") or ""),
        "source_name": str(item.get("source_name") or "tonghuashun"),
    }
    fingerprint = json.dumps({"text": text, "metadata": metadata}, sort_keys=True, ensure_ascii=False)
    metadata["content_hash"] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return IndexRecord(article_id=article_id, text=text, metadata=metadata)


def open_news_collection(path: str | Path | None = None, name: str | None = None):
    """打开独立于示例文档的新闻持久化集合。"""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before building the news index") from exc
    directory = Path(path or os.getenv("NEWS_CHROMA_DIR") or DEFAULT_INDEX_DIR)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory = directory.resolve()
    collection_name = name or os.getenv("NEWS_CHROMA_COLLECTION") or DEFAULT_COLLECTION
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"embedding_model": settings.embedding_model}
    )
    indexed_model = (collection.metadata or {}).get("embedding_model")
    if indexed_model and indexed_model != settings.embedding_model:
        raise ValueError(
            f"Collection was built with {indexed_model}; use a new collection for {settings.embedding_model}"
        )
    return collection


@lru_cache(maxsize=1)
def get_embedding_model():
    """按需加载项目的本地多语言嵌入模型。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before building the news index") from exc
    model_name = settings.embedding_model
    local_model = Path(model_name)
    if not local_model.is_absolute() and (PROJECT_ROOT / local_model).exists():
        model_name = str(PROJECT_ROOT / local_model)
    return HuggingFaceEmbeddings(model_name=model_name)


def _write_batch(
    records: list[IndexRecord],
    collection: Any,
    embed_texts: Callable[[list[str]], list[list[float]]],
    stats: SyncStats,
) -> None:
    existing = collection.get(ids=[record.article_id for record in records], include=["metadatas"])
    prior = dict(zip(existing["ids"], existing["metadatas"]))
    changed: list[IndexRecord] = []
    for record in records:
        old = prior.get(record.article_id)
        if old and old.get("content_hash") == record.metadata["content_hash"]:
            stats.unchanged += 1
        else:
            changed.append(record)
    if not changed:
        return
    embeddings = embed_texts([record.text for record in changed])
    if len(embeddings) != len(changed):
        raise ValueError("Embedding model returned an unexpected number of vectors")
    collection.upsert(
        ids=[record.article_id for record in changed],
        embeddings=embeddings,
        documents=[record.text for record in changed],
        metadatas=[record.metadata for record in changed],
    )
    stats.inserted += sum(record.article_id not in prior for record in changed)
    stats.updated += sum(record.article_id in prior for record in changed)


def upsert_news_items(
    items: list[dict[str, Any]],
    collection: Any,
    embed_texts: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int = 32,
) -> SyncStats:
    """把查询得到的候选新闻写入向量缓存，并跳过未变化的文章。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stats = SyncStats()
    batch: list[IndexRecord] = []
    for item in items:
        stats.fetched += 1
        record = prepare_record(item)
        if record is None:
            stats.invalid += 1
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            _write_batch(batch, collection, embed_texts, stats)
            batch.clear()
    if batch:
        _write_batch(batch, collection, embed_texts, stats)
    return stats


def sync_recent_news(
    client: NewsClient,
    collection: Any,
    embed_texts: Callable[[list[str]], list[list[float]]],
    *,
    max_items: int = 1000,
    page_size: int = 100,
    batch_size: int = 32,
) -> SyncStats:
    """索引最新文章；重复运行会跳过未变化的 ID，并可安全地继续执行。

    此操作只处理最新的 max_items 篇文章，而非全部历史归档。
    运行失败后可以安全地从头重试。
    """
    if max_items < 1 or page_size < 1 or page_size > 100 or batch_size < 1:
        raise ValueError("max_items and batch_size must be positive; page_size must be 1..100")
    stats = SyncStats()
    batch: list[IndexRecord] = []
    for item in client.iter_news(page_size=page_size, max_items=max_items, order="desc"):
        stats.fetched += 1
        record = prepare_record(item)
        if record is None:
            stats.invalid += 1
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            _write_batch(batch, collection, embed_texts, stats)
            batch.clear()
    if batch:
        _write_batch(batch, collection, embed_texts, stats)
    return stats


def search_news(
    query: str,
    collection: Any,
    embed_query: Callable[[str], list[float]],
    *,
    k: int = 5,
    section: str = "",
    article_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """搜索本机新闻索引；需要全文时使用 NewsClient.article。"""
    if not query.strip() or k < 1:
        raise ValueError("query must not be blank and k must be positive")
    options: dict[str, Any] = {
        "query_embeddings": [embed_query(query)],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    filters = []
    if section:
        filters.append({"section": section})
    if article_ids:
        filters.append({"article_id": {"$in": article_ids}})
    if len(filters) == 1:
        options["where"] = filters[0]
    elif filters:
        options["where"] = {"$and": filters}
    result = collection.query(**options)
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]
    return [
        {"article_id": article_id, "text": document, "metadata": metadata, "distance": distance}
        for article_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query a local index of server news")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Inspect API data without creating an index")
    preview.add_argument("--max-items", type=int, default=20)
    sync = commands.add_parser("sync", help="Index the newest articles incrementally")
    sync.add_argument("--max-items", type=int, default=1000)
    sync.add_argument("--batch-size", type=int, default=32)
    search = commands.add_parser("search", help="Search the local news index")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "preview":
        if args.max_items < 1:
            parser.error("--max-items must be positive")
        client = NewsClient()
        fetched = valid = 0
        for item in client.iter_news(max_items=args.max_items):
            fetched += 1
            valid += prepare_record(item) is not None
        print(f"API preview: {fetched} fetched, {valid} indexable; no files changed")
        return

    collection = open_news_collection()
    model = get_embedding_model()
    if args.command == "sync":
        stats = sync_recent_news(
            NewsClient(), collection, model.embed_documents,
            max_items=args.max_items, batch_size=args.batch_size,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False))
    else:
        matches = search_news(args.query, collection, model.embed_query, k=args.k)
        for match in matches:
            metadata = match["metadata"] or {}
            print(f"{match['article_id']} | {metadata.get('published_at', '')} | {metadata.get('title', '')}")


if __name__ == "__main__":
    main()
