"""用本机假代理返回 503，回归验证预热和同步/异步推理都走正确连接。"""

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx
import pytest

from agentic_rag import cli
from agentic_rag.graph import chains
from agentic_rag.ollama_connection import ollama_client_kwargs


@contextmanager
def local_server(*, reject=False, usage=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.respond()

        def do_GET(self):
            self.respond()

        def respond(self):
            self.server.requests.append(self.path)
            if reject:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = {
                "model": "test",
                "message": {"role": "assistant", "content": "本机直连成功"},
                "response": "",
                "done": True,
            }
            if usage:
                data.update(prompt_eval_count=23, eval_count=5, prompt_eval_cached_count=11)
            content = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("url", [
    "http://localhost:11434", "http://LOCALHOST:11434",
    "http://127.0.0.1:11434", "http://127.0.0.2:11434",
    "http://[::1]:11434", "localhost:11434",
])
def test_loopback_uses_direct_connection(url):
    assert ollama_client_kwargs(url) == {"trust_env": False}


@pytest.mark.parametrize("url", [
    "https://ollama.example.com", "http://192.168.1.20:11434",
    "http://localhost.example.com:11434",
])
def test_remote_keeps_network_configuration(url):
    assert ollama_client_kwargs(url) == {}


def test_proxy_503_cannot_break_warmup_or_sync_async_inference(monkeypatch):
    with local_server(reject=True) as (proxy, proxy_url), local_server() as (ollama, base_url):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")

        # 先验证测试环境确实会复现旧故障，而不是意外绕过了假代理。
        with httpx.Client(timeout=5) as client:
            assert client.get(base_url + "/api/version").status_code == 503
        assert len(proxy.requests) == 1

        test_settings = replace(cli.settings, ollama_base_url=base_url, llm_model="test")
        monkeypatch.setattr(cli, "settings", test_settings)
        monkeypatch.setattr(chains, "settings", test_settings)
        chains.get_llm.cache_clear()
        model = None
        try:
            assert cli.warm_up_model()
            model = chains.get_llm()
            assert model.invoke("测试").content == "本机直连成功"

            async def async_request():
                try:
                    result = await model.ainvoke("测试")
                    assert result.content == "本机直连成功"
                finally:
                    await model._async_client._client.aclose()

            asyncio.run(async_request())
            assert ollama.requests == ["/api/generate", "/api/chat", "/api/chat"]
            assert len(proxy.requests) == 1
        finally:
            if model is not None:
                model._client._client.close()
            chains.get_llm.cache_clear()


def test_real_ollama_adapter_retains_final_stream_usage():
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from agentic_rag.token_usage import UsageLedger, usage_session

    with local_server(usage=True) as (server, base_url):
        model = ChatOllama(model="test", base_url=base_url, reasoning=False,
                           client_kwargs={"trust_env": False})
        ledger = UsageLedger()
        try:
            with usage_session(ledger):
                answer = chains._with_model_retry(model | StrOutputParser()).invoke("测试")
            assert answer == "本机直连成功"
            assert server.requests == ["/api/chat"]
            stats = ledger.report()["current"]
            assert stats["input_tokens"] == 23 and stats["output_tokens"] == 5
            assert stats["total_tokens"] == 28 and stats["calls"] == 1
            # 老版 SDK 丢弃新增字段；新版若透传就必须使用精确值，不能猜成 0。
            assert stats["cached_input_tokens"] in (None, 11)
            assert stats["cached_unknown"] == (1 if stats["cached_input_tokens"] is None else 0)
            assert stats["reasoning_unknown"] == 1
        finally:
            model._client._client.close()
