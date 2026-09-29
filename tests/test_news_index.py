"""增量新闻索引与检索的离线测试。"""

import unittest

from agentic_rag.news_index import (
    prepare_record,
    search_news,
    sync_recent_news,
    upsert_news_items,
)


def article(article_id, summary="摘要"):
    return {
        "article_id": article_id,
        "title": f"新闻 {article_id}",
        "summary": summary,
        "published_at": "2026-09-29T00:00:00Z",
        "section": "财经",
        "canonical_url": f"https://example.com/{article_id}",
        "source_name": "tonghuashun",
    }


class FakeClient:
    def __init__(self, items):
        self.items = items

    def iter_news(self, **kwargs):
        yield from self.items[: kwargs["max_items"]]


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.upsert_calls = 0

    def get(self, *, ids, include):
        found = [article_id for article_id in ids if article_id in self.rows]
        return {"ids": found, "metadatas": [self.rows[article_id]["metadata"] for article_id in found]}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upsert_calls += 1
        for article_id, vector, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[article_id] = {"vector": vector, "document": document, "metadata": metadata}

    def query(self, **kwargs):
        self.query_options = kwargs
        return {
            "ids": [["one"]],
            "documents": [["新闻 one\n摘要"]],
            "metadatas": [[{"title": "新闻 one"}]],
            "distances": [[0.1]],
        }


class NewsIndexTests(unittest.TestCase):
    def test_prepare_record_uses_title_and_summary_only(self):
        item = article("one")
        item["content"] = "很长的全文不应写入第一版索引"
        record = prepare_record(item)
        self.assertEqual(record.text, "新闻 one\n摘要")
        self.assertNotIn(item["content"], record.text)
        self.assertEqual(record.metadata["article_id"], "one")

    def test_repeat_sync_skips_unchanged_and_updates_changed(self):
        source = FakeClient([article("one"), article("two")])
        collection = FakeCollection()
        calls = []

        def embed(texts):
            calls.append(texts)
            return [[float(len(text))] for text in texts]

        first = sync_recent_news(source, collection, embed, max_items=2, batch_size=2)
        self.assertEqual((first.fetched, first.inserted, first.updated), (2, 2, 0))
        self.assertEqual(len(collection.rows), 2)

        second = sync_recent_news(source, collection, embed, max_items=2, batch_size=2)
        self.assertEqual((second.inserted, second.unchanged), (0, 2))
        self.assertEqual(len(calls), 1)

        source.items[1] = article("two", summary="修改后的摘要")
        third = sync_recent_news(source, collection, embed, max_items=2, batch_size=2)
        self.assertEqual((third.updated, third.unchanged), (1, 1))
        self.assertEqual(len(collection.rows), 2)

    def test_failed_batch_can_be_retried_without_duplicate_ids(self):
        source = FakeClient([article("one"), article("two")])
        collection = FakeCollection()
        calls = 0

        def fail_second_batch(texts):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("temporary embedding error")
            return [[1.0] for _ in texts]

        with self.assertRaisesRegex(RuntimeError, "temporary embedding error"):
            sync_recent_news(source, collection, fail_second_batch, max_items=2, batch_size=1)
        self.assertEqual(set(collection.rows), {"one"})

        recovered = sync_recent_news(
            source, collection, lambda texts: [[1.0] for _ in texts],
            max_items=2, batch_size=1,
        )
        self.assertEqual((recovered.inserted, recovered.unchanged), (1, 1))
        self.assertEqual(set(collection.rows), {"one", "two"})

    def test_invalid_article_is_skipped(self):
        collection = FakeCollection()
        stats = sync_recent_news(
            FakeClient([{"article_id": "missing-title"}]), collection,
            lambda texts: [[1.0] for _ in texts], max_items=1,
        )
        self.assertEqual((stats.fetched, stats.invalid), (1, 1))
        self.assertEqual(collection.rows, {})

    def test_search_returns_ids_text_and_metadata(self):
        collection = FakeCollection()
        matches = search_news(
            "市场", collection, lambda query: [0.5], k=3,
            section="财经", article_ids=["one", "two"],
        )
        self.assertEqual(matches[0]["article_id"], "one")
        self.assertEqual(matches[0]["text"], "新闻 one\n摘要")
        self.assertEqual(
            collection.query_options["where"],
            {"$and": [{"section": "财经"}, {"article_id": {"$in": ["one", "two"]}}]},
        )

    def test_query_candidates_use_the_same_incremental_upsert(self):
        collection = FakeCollection()
        first = upsert_news_items(
            [article("one")], collection, lambda texts: [[1.0] for _ in texts]
        )
        second = upsert_news_items(
            [article("one")], collection, lambda texts: [[1.0] for _ in texts]
        )
        self.assertEqual((first.inserted, second.unchanged), (1, 1))


if __name__ == "__main__":
    unittest.main()
