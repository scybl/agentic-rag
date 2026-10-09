"""在新闻容器内验证已挂载服务的分页与错误协议；不访问数据库或真实密钥。"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


sys.path.insert(0, "/service")
api = sys.modules.get("verified_news_api")
if api is None:
    spec = importlib.util.spec_from_file_location("verified_news_api", os.getenv("NEWS_API_TEST_SOURCE", "/service/news_api.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)


def handler():
    target = object.__new__(api.Handler)
    target.headers = SimpleNamespace(get_all=lambda *_: ["test-key"])
    target.server = SimpleNamespace(api_keys=SimpleNamespace(authorized=lambda _: True),
                                    slots=threading.BoundedSemaphore(1))
    target.path = "/v1/news?q=test&limit=20"
    target.send_json = Mock()
    return target


class ServiceContractTests(unittest.TestCase):
    def test_pagination_uses_existing_index_and_preserves_filters(self):
        target = handler()
        rows = [{"article_id": str(i), "published_at": "2026-09-30T00:00:00Z"} for i in range(21)]
        cursor = Mock()
        cursor.hint.return_value = cursor
        cursor.sort.return_value = cursor
        cursor.limit.return_value = cursor
        cursor.max_time_ms.return_value = rows
        collection = Mock()
        collection.find.return_value = cursor
        target.server.collection = collection
        target.dispatch()
        cursor.hint.assert_called_once_with("idx_agent_source_published_article")
        cursor.limit.assert_called_once_with(21)
        cursor.max_time_ms.assert_called_once_with(8000)
        query = collection.find.call_args.args[0]
        self.assertNotIn("$gte", json.dumps(query))
        status, result = target.send_json.call_args.args
        self.assertEqual((status, len(result["items"]), result["has_more"]), (200, 20, True))
        scope = api.parse_request({"q": ["test"], "limit": ["20"]})[-1]
        self.assertEqual(result["next_cursor"], api.encode_cursor(rows[19], scope))

    def test_error_types_are_distinct_and_slots_are_released(self):
        cases = [(api.ExecutionTimeout("PRIVATE"), 503, "query_timeout", False),
                 (api.ConnectionFailure("PRIVATE"), 503, "database_unavailable", True),
                 (RuntimeError("PRIVATE"), 500, "internal_error", False)]
        for exception, status, code, retryable in cases:
            with self.subTest(code=code):
                target = handler()
                target.dispatch = Mock(side_effect=exception)
                output = io.StringIO()
                with redirect_stdout(output):
                    target.do_GET()
                actual_status, payload = target.send_json.call_args.args
                self.assertEqual((actual_status, payload["error_code"], payload["retryable"]),
                                 (status, code, retryable))
                self.assertNotIn("PRIVATE", output.getvalue() + json.dumps(payload))
                self.assertEqual(json.loads(output.getvalue())["request_id"], target.request_id)
                self.assertTrue(target.server.slots.acquire(blocking=False))

    def test_busy_is_retryable_without_releasing_an_unowned_slot(self):
        target = handler()
        target.server.slots.acquire()
        target.do_GET()
        status, payload = target.send_json.call_args.args
        self.assertEqual((status, payload["error_code"], payload["retryable"]), (503, "busy", True))
        self.assertFalse(target.server.slots.acquire(blocking=False))

    def test_cursor_cannot_be_reused_after_filters_change(self):
        scope = api.parse_request({"q": ["original"]})[-1]
        cursor = api.encode_cursor({"article_id": "a", "published_at": "2026-09-30T00:00:00Z"}, scope)
        with self.assertRaisesRegex(ValueError, "filters changed"):
            api.parse_request({"q": ["changed"], "cursor": [cursor]})

    def test_response_header_and_error_body_share_correlation_id(self):
        target = handler()
        target.send_response = Mock()
        target.send_header = Mock()
        target.end_headers = Mock()
        target.wfile = io.BytesIO()
        api.Handler.send_json(target, 503, {"error_code": "busy", "retryable": True})
        payload = json.loads(target.wfile.getvalue())
        target.send_header.assert_any_call("X-Request-ID", payload["request_id"])
        target.send_header.assert_any_call("Retry-After", "1")
        self.assertEqual(len(payload["request_id"]), 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
