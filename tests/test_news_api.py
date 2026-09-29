"""只读新闻 API 客户端的离线测试。"""

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from agentic_rag.news_api import NewsAPIError, NewsClient


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode("utf-8"))


class NewsClientTests(unittest.TestCase):
    def make_client(self, *responses):
        client = NewsClient(api_key="test-secret", base_url="https://example.com:3001")
        client._opener = FakeOpener(*responses)
        return client

    def test_requires_key_and_https(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "NEWS_API_KEY"):
                NewsClient(base_url="https://example.com")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            NewsClient(api_key="test-secret", base_url="http://example.com:3001")
        with self.assertRaises(ValueError):
            NewsClient(api_key="test-secret", base_url="https://example.com/?key=secret")

    def test_search_sends_key_in_header_and_filters_in_query(self):
        client = self.make_client({"items": [], "has_more": False})
        page = client.search(q="人工智能", section="财经", limit=3)
        self.assertEqual(page["items"], [])
        request, timeout = client._opener.requests[0]
        self.assertEqual(timeout, 15)
        self.assertEqual(request.get_header("X-api-key"), "test-secret")
        self.assertNotIn("test-secret", request.full_url)
        parsed = urlsplit(request.full_url)
        self.assertEqual(parsed.path, "/v1/news")
        self.assertEqual(parse_qs(parsed.query)["q"], ["人工智能"])
        self.assertEqual(parse_qs(parsed.query)["section"], ["财经"])
        self.assertEqual(parse_qs(parsed.query)["limit"], ["3"])

    def test_empty_dates_are_not_sent_and_trace_contains_only_query_fields(self):
        client = self.make_client({"items": [], "has_more": False})
        events = []
        list(client.iter_news(q="生猪", page_size=20, max_items=30, on_event=events.append))
        params = parse_qs(urlsplit(client._opener.requests[0][0].full_url).query)
        self.assertNotIn("start", params)
        self.assertNotIn("end", params)
        self.assertEqual(events[0]["kind"], "news_request")
        self.assertEqual(events[0]["limit"], 20)
        self.assertEqual(events[0]["max_items"], 30)
        self.assertEqual(events[1]["count"], 0)
        self.assertNotIn("test-secret", str(events))

    def test_article_uses_encoded_id_and_returns_item(self):
        client = self.make_client({"item": {"article_id": "a/b", "content": "正文"}})
        article = client.article("a/b")
        self.assertEqual(article["content"], "正文")
        self.assertTrue(client._opener.requests[0][0].full_url.endswith("/v1/news/a%2Fb"))

    def test_iter_news_follows_cursor_and_respects_cap(self):
        client = self.make_client(
            {"items": [{"article_id": "1"}, {"article_id": "2"}], "has_more": True, "next_cursor": "next"},
            {"items": [{"article_id": "3"}], "has_more": False},
        )
        items = list(client.iter_news(page_size=2, max_items=3, q="市场"))
        self.assertEqual([item["article_id"] for item in items], ["1", "2", "3"])
        second_query = parse_qs(urlsplit(client._opener.requests[1][0].full_url).query)
        self.assertEqual(second_query["cursor"], ["next"])
        self.assertEqual(second_query["limit"], ["1"])
        self.assertEqual(second_query["q"], ["市场"])

    def test_repeated_cursor_is_rejected(self):
        client = self.make_client(
            {"items": [{"article_id": "1"}], "has_more": True, "next_cursor": "same"},
            {"items": [{"article_id": "2"}], "has_more": True, "next_cursor": "same"},
        )
        with self.assertRaisesRegex(NewsAPIError, "repeated pagination cursor"):
            list(client.iter_news(page_size=1, max_items=None))

    def test_http_error_exposes_status_without_key(self):
        client = self.make_client(HTTPError("https://example.com/health", 401, "Unauthorized", {}, None))
        with self.assertRaises(NewsAPIError) as caught:
            client.health()
        self.assertEqual(caught.exception.status_code, 401)
        self.assertNotIn("test-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
