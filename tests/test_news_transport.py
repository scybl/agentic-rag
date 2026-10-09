"""真实HTTP连接验证认证密钥不会随重定向转发；只使用本机临时服务和假密钥。"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from agentic_rag.news_api import NewsAPIError, NewsClient


@contextmanager
def server(handler):
    instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(2)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_never_forwards_news_credentials(status):
    received = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("X-API-Key"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_):
            pass

    with server(Destination) as target:
        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(status)
                self.send_header("Location", target + "/health")
                self.end_headers()

            def log_message(self, *_):
                pass

        with server(Redirect) as origin:
            client = NewsClient(api_key="dummy-key-never-publish", base_url=origin, timeout=2)
            with pytest.raises(NewsAPIError) as caught:
                client.health()
            assert caught.value.status_code == status
            assert not caught.value.retryable
            assert "dummy-key" not in str(caught.value)
            assert received == []
