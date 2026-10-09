"""只读新闻 API 客户端的离线测试。"""

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
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

    def test_transient_503_retries_same_page_and_emits_safe_progress(self):
        events = []
        client = self.make_client(HTTPError("https://example.com", 503, "busy", {}, None),
                                  {"items": [{"article_id": "one"}], "has_more": False})
        client.on_retry = events.append
        with patch("agentic_rag.news_api.time.sleep") as sleep:
            items = list(client.iter_news(q="特朗普 泽连斯基", page_size=20))
        self.assertEqual([item["article_id"] for item in items], ["one"])
        self.assertEqual(len(client._opener.requests), 2)
        self.assertEqual(client._opener.requests[0][0].full_url, client._opener.requests[1][0].full_url)
        self.assertEqual(events[0]["attempt"], 2)
        self.assertNotIn("test-secret", str(events))
        sleep.assert_called_once_with(0.5)

    def test_retries_have_a_hard_cap_and_do_not_retry_bad_auth_or_connection(self):
        errors = [HTTPError("https://example.com", 503, "busy", {}, None) for _ in range(3)]
        client = self.make_client(*errors)
        with patch("agentic_rag.news_api.time.sleep") as sleep, self.assertRaises(NewsAPIError):
            client.health()
        self.assertEqual(len(client._opener.requests), 3)
        self.assertEqual(sleep.call_count, 2)
        for error in (HTTPError("https://example.com", 403, "denied", {}, None),
                      URLError(PermissionError("SECRET")), TimeoutError("SECRET")):
            client = self.make_client(error)
            with patch("agentic_rag.news_api.time.sleep") as sleep, self.assertRaises(NewsAPIError) as caught:
                client.health()
            sleep.assert_not_called()
            self.assertNotIn("SECRET", str(caught.exception))

    def test_long_retry_after_does_not_hammer_rate_limited_service(self):
        client = self.make_client(HTTPError("https://example.com", 429, "limited", {"Retry-After": "120"}, None))
        with patch("agentic_rag.news_api.time.sleep") as sleep, self.assertRaises(NewsAPIError):
            client.health()
        sleep.assert_not_called()

    def test_observed_query_deadline_is_not_retried_or_silently_narrowed(self):
        body = io.BytesIO(json.dumps({"error": "Query unavailable or exceeded 8 seconds; narrow the date range"}).encode())
        error = HTTPError("https://example.com", 503, "unavailable", {}, body)
        client = self.make_client(error)
        with patch("agentic_rag.news_api.time.sleep") as sleep, self.assertRaises(NewsAPIError) as caught:
            client.search(q="特朗普 泽连斯基", cursor="next", limit=20)
        self.assertFalse(caught.exception.retryable)
        self.assertIn("未自动缩小时间范围", str(caught.exception))
        self.assertEqual(len(client._opener.requests), 1)
        params = parse_qs(urlsplit(client._opener.requests[0][0].full_url).query)
        self.assertNotIn("start", params)
        self.assertNotIn("end", params)
        sleep.assert_not_called()
        self.assertTrue(body.closed)

    def test_structured_timeout_and_internal_failure_do_not_retry_or_expose_body(self):
        for status, code in ((503, "query_timeout"), (500, "internal_error")):
            with self.subTest(code=code):
                request_id = "a" * 32
                body = io.BytesIO(json.dumps({"error_code": code, "error": "SECRET",
                                             "retryable": False, "request_id": request_id}).encode())
                client = self.make_client(HTTPError("https://example.com", status, "failed", {}, body))
                with patch("agentic_rag.news_api.time.sleep") as sleep, self.assertRaises(NewsAPIError) as caught:
                    client.health()
                error = caught.exception
                self.assertEqual((error.error_code, error.request_id, error.retryable), (code, request_id, False))
                self.assertIn(request_id, str(error))
                self.assertNotIn("SECRET", str(error))
                sleep.assert_not_called()

    def test_busy_retry_preserves_server_correlation_id(self):
        events = []
        body = io.BytesIO(json.dumps({"error_code": "busy", "retryable": True}).encode())
        client = self.make_client(HTTPError("https://example.com", 503, "busy",
                                           {"X-Request-ID": "b" * 32, "Retry-After": "1"}, body),
                                  {"ok": True})
        client.on_retry = events.append
        with patch("agentic_rag.news_api.time.sleep") as sleep:
            self.assertTrue(client.health()["ok"])
        sleep.assert_called_once_with(1)
        self.assertEqual((events[0]["error_code"], events[0]["request_id"]), ("busy", "b" * 32))

    def test_successful_page_correlation_id_is_retained_in_audit_event(self):
        from unittest.mock import Mock
        response = io.BytesIO(json.dumps({"items": [], "has_more": False}).encode())
        response.headers = {"X-Request-ID": "c" * 32}
        client = self.make_client()
        client._opener = Mock()
        client._opener.open.return_value = response
        events = []
        self.assertEqual(list(client.iter_news(q="test", on_event=events.append)), [])
        self.assertEqual(events[-1]["request_id"], "c" * 32)

    def test_unknown_error_fields_are_not_copied_into_logs(self):
        body = io.BytesIO(json.dumps({"error_code": ["SECRET"], "request_id": "PRIVATE\nFORGED",
                                     "error": "password=SECRET"}).encode())
        client = self.make_client(HTTPError("https://example.com", 400, "bad", {}, body))
        with self.assertRaises(NewsAPIError) as caught:
            client.health()
        self.assertEqual((caught.exception.error_code, caught.exception.request_id), ("", ""))
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn("FORGED", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
