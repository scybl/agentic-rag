"""预热连接、临时服务错误、请求超时和资源关闭的离线回归。"""

import socket
from dataclasses import replace

import httpx
import pytest
from ollama import ResponseError

from agentic_rag import cli
from agentic_rag.ollama_connection import safe_endpoint, warmup_failure
from agentic_rag.token_usage import UsageLedger


class Client:
    def __init__(self, failures):
        self.failures = iter(failures)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        error = next(self.failures, None)
        if error is not None:
            raise error
        return {"done": True}


@pytest.mark.parametrize("error,label", [
    (ConnectionError("PRIVATE"), "尚未确认模型开始加载"),
    (httpx.ConnectTimeout("PRIVATE"), "无法建立"),
    (httpx.ReadTimeout("PRIVATE"), "服务端可能仍在加载"),
    (ResponseError("PRIVATE", 404), "ollama list"),
    (ResponseError("PRIVATE", 401), "认证"),
    (ResponseError("PRIVATE", 403), "权限"),
    (ResponseError("PRIVATE", 400), "HTTP 400"),
    (ResponseError("PRIVATE", 500), "HTTP 500"),
    (ValueError("PRIVATE"), "ValueError"),
])
def test_nonretryable_failure_is_one_attempt_and_usage_remains_unknown(error, label, monkeypatch, capsys):
    monkeypatch.setattr(cli, "settings", replace(cli.settings, ollama_base_url="http://localhost:11434",
                                               ollama_warmup_attempts=10))
    client, sleeps, ledger = Client([error]), [], UsageLedger()
    assert not cli.warm_up_model(client=client, sleep_fn=sleeps.append, session_ledger=ledger)
    text = capsys.readouterr().out
    assert label in text and "PRIVATE" not in text and "模型暂未就绪" not in text
    assert client.calls == 1 and sleeps == []
    usage = ledger.report()["cumulative"]  # 程序账本中的预热是独立 session，不是当前问题。
    assert usage["calls"] == 1 and usage["failed"] == 1
    assert usage["unknown"] == 1


@pytest.mark.parametrize("code", [429, 502, 503, 504])
def test_transient_http_can_recover_without_hiding_status(code, monkeypatch, capsys):
    monkeypatch.setattr(cli, "settings", replace(cli.settings, ollama_warmup_attempts=3,
                                               ollama_warmup_retry_seconds=0.25))
    client, sleeps = Client([ResponseError("PRIVATE", code)]), []
    assert cli.warm_up_model(client=client, sleep_fn=sleeps.append)
    assert sleeps == [0.25] and client.calls == 2
    assert f"HTTP {code}" in capsys.readouterr().out


def test_retry_exhaustion_has_no_extra_sleep(monkeypatch, capsys):
    monkeypatch.setattr(cli, "settings", replace(cli.settings, ollama_warmup_attempts=2))
    client, sleeps = Client([ResponseError("busy", 503)] * 2), []
    assert not cli.warm_up_model(client=client, sleep_fn=sleeps.append)
    assert client.calls == 2 and len(sleeps) == 1
    assert "已达到预热尝试上限" in capsys.readouterr().out


def test_actual_sdk_closed_local_port_fails_fast_and_closes_client(monkeypatch, capsys):
    # 占用但不监听，避免另一个服务抢占端口；SDK 会将 httpx.ConnectError 转成 ConnectionError。
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        monkeypatch.setattr(cli, "settings", replace(cli.settings,
            ollama_base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}", llm_request_timeout=2))
        actual, created = cli.Client, []
        def create(**kwargs):
            instance = actual(**kwargs)
            created.append(instance)
            return instance
        monkeypatch.setattr(cli, "Client", create)
        sleeps = []
        assert not cli.warm_up_model(sleep_fn=sleeps.append)
        assert sleeps == [] and len(created) == 1
        assert created[0]._client.is_closed
        assert created[0]._client.timeout.connect == 2
        assert created[0]._client.timeout.read == 2
    assert "ollama serve" in capsys.readouterr().out


def test_owned_client_is_closed_after_success(monkeypatch):
    client = Client([])
    closed = []
    class ManagedClient:
        def __init__(self, **kwargs):
            assert kwargs["timeout"].connect <= 5
        def __enter__(self):
            return client
        def __exit__(self, *args):
            closed.append(True)
    monkeypatch.setattr(cli, "Client", ManagedClient)
    assert cli.warm_up_model()
    assert closed == [True] and client.calls == 1


def test_bad_client_configuration_has_no_traceback_or_phantom_usage(monkeypatch, capsys):
    def invalid(**kwargs):
        raise ValueError("PRIVATE")
    monkeypatch.setattr(cli, "Client", invalid)
    ledger = UsageLedger()
    assert not cli.warm_up_model(session_ledger=ledger)
    text = capsys.readouterr().out
    assert "初始化失败" in text and "PRIVATE" not in text
    assert ledger.report()["cumulative"]["calls"] == 0


def test_cli_failed_warmup_does_not_send_second_unload_request(monkeypatch):
    from agentic_rag.research import service
    monkeypatch.setattr(cli, "settings", replace(cli.settings, knowledge_watch_enabled=False,
                                               ollama_warmup_enabled=True))
    monkeypatch.setattr("sys.argv", ["agentic-rag"])
    monkeypatch.setattr(service, "store", lambda: object())
    monkeypatch.setattr(cli, "warm_up_model", lambda **kwargs: False)
    def unexpected(**kwargs):
        pytest.fail("失败的预热后不应额外卸载或创建研究图")
    monkeypatch.setattr(cli, "unload_model", unexpected)
    monkeypatch.setattr(cli, "build_graph", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1


def test_unload_timeout_does_not_hide_original_failure(monkeypatch, capsys):
    client = Client([httpx.ReadTimeout("PRIVATE")])
    assert not cli.unload_model(client=client, verbose=True)
    text = capsys.readouterr().out
    assert "ReadTimeout" in text and "PRIVATE" not in text


def test_remote_connection_failure_does_not_prescribe_local_service():
    retry, text = warmup_failure(ConnectionError(), "https://user:SECRET@remote.example:123/private?key=SECRET")
    assert not retry and "远程" in text and "ollama serve" not in text
    assert "SECRET" not in text and "/private" not in text
    assert "https://remote.example:123" in text


@pytest.mark.parametrize("url,expected", [
    ("http://[::1]:11434", "http://[::1]:11434"),
    ("localhost:11434", "http://localhost:11434"),
    ("http://localhost:bad", "（地址格式无效，请检查 OLLAMA_BASE_URL）"),
])
def test_endpoint_format(url, expected):
    assert safe_endpoint(url) == expected
