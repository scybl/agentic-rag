"""带证据版本的分析缓存测试。"""

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from agentic_rag.analysis_cache import AnalysisCache, build_cache_key


def evidence(source, text):
    return Document(
        page_content=text,
        metadata={"source": source, "source_type": "news_api"},
    )


class AnalysisCacheTests(unittest.TestCase):
    def test_key_is_order_independent_but_changes_with_evidence(self):
        first = evidence("one", "证据一")
        second = evidence("two", "证据二")
        key_a, _, _ = build_cache_key(
            "这个问题", [first, second], model="test", prompt_version="v1"
        )
        key_b, _, _ = build_cache_key(
            "  这个问题 ", [second, first], model="test", prompt_version="v1"
        )
        key_c, _, _ = build_cache_key(
            "这个问题", [first, evidence("two", "更新后的证据")],
            model="test", prompt_version="v1",
        )
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_cache_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisCache(Path(directory) / "answers.sqlite3")
            cache.put(
                "key",
                question="问题",
                answer="答案",
                evidence_hash="hash",
                source_ids=["one"],
                model="test",
                prompt_version="v1",
                ttl_seconds=10,
                now=100,
            )
            self.assertEqual(cache.get("key", now=109), "答案")
            self.assertIsNone(cache.get("key", now=110))

    def test_blank_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisCache(Path(directory) / "answers.sqlite3")
            with self.assertRaisesRegex(ValueError, "must not be blank"):
                cache.put(
                    "key", question="问题", answer="  ", evidence_hash="hash",
                    source_ids=["one"], model="test", prompt_version="v1",
                    ttl_seconds=10,
                )


if __name__ == "__main__":
    unittest.main()
