"""实开本地 TLS 套接字：慢握手不能堵塞后续合法连接，无外部网络。"""

import importlib.util
import socket
import ssl
import threading
import time
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("tls_server", Path(__file__).resolve().parents[1] / "ops/news_api/tls_server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def context(tmp_path):
    openssl = shutil.which("openssl")
    bundled = Path(sys.prefix) / "Library/bin/openssl.exe"
    if not openssl and bundled.exists():
        openssl = str(bundled)
    if not openssl:
        pytest.skip("TLS 实机测试需要 openssl 生成一次性测试证书")
    cert, private = tmp_path / "cert.pem", tmp_path / "key.pem"
    config = tmp_path / "openssl.cnf"
    config.write_text("[req]\ndistinguished_name=dn\n[dn]\n", encoding="ascii")
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-config", str(config), "-subj", "/CN=localhost", "-keyout", str(private), "-out", str(cert)],
                   check=True, capture_output=True, timeout=15)
    result = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    result.load_cert_chain(cert, private)
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *_):
        pass


def request(address, timeout=1):
    context = ssl._create_unverified_context()  # 仅此测试的临时自签名证书；生产客户端仍验证 TLS。
    with socket.create_connection(address, timeout=timeout) as raw:
        with context.wrap_socket(raw, server_hostname="localhost") as sock:
            sock.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            return sock.recv(4096)


def test_real_slow_handshake_reproduces_old_accept_blocking(context):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    slow = socket.create_connection(server.server_address, timeout=1)
    try:
        with pytest.raises(TimeoutError):
            request(server.server_address, timeout=.2)
    finally:
        slow.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_slow_handshake_no_longer_blocks_good_connection(context):
    server = module.BoundedTLSHTTPServer(("127.0.0.1", 0), Handler, ssl_context=context, connection_timeout=.3, max_connections=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    slow = socket.create_connection(server.server_address, timeout=1)
    try:
        assert b"200" in request(server.server_address)
        time.sleep(.4)
        assert slow.recv(1) == b""  # 慢握手已过期，不是永久占着一个线程。
        assert b"200" in request(server.server_address)
    finally:
        slow.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_connection_limit_and_recovery_after_bad_handshake(context):
    entered = threading.Event()
    class Observed(module.BoundedTLSHTTPServer):
        def process_request_thread(self, request, address):
            entered.set()
            super().process_request_thread(request, address)
    server = Observed(("127.0.0.1", 0), Handler, ssl_context=context, connection_timeout=.3, max_connections=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    slow = socket.create_connection(server.server_address, timeout=1)
    try:
        assert entered.wait(1)
        with pytest.raises((OSError, ssl.SSLError)):
            request(server.server_address, timeout=.2)
        slow.close()
        time.sleep(.4)
        assert b"200" in request(server.server_address)
    finally:
        slow.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_deployment_transform_is_exact_and_idempotent():
    location = Path(__file__).resolve().parents[1] / "ops/news_api/install_tls.py"
    spec = importlib.util.spec_from_file_location("install_tls", location)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    source = ("def main():\n"
        "    def make_server(port):\n        server = ThreadingHTTPServer(('0.0.0.0', port), Handler)\n"
        "        return server\n    public = make_server(3001)\n"
        "    public.socket = tls.wrap_socket(public.socket, server_side=True)\n")
    helper = location.with_name("tls_server.py").read_text(encoding="utf-8")
    result = installer.patched(source, helper)
    assert "public = make_server(3001, tls)" in result
    assert "do_handshake_on_connect=False" in result
    assert installer.patched(result, helper) == result
    with pytest.raises(ValueError):
        installer.patched(source.replace("3001", "3002"), helper)


def test_installer_dry_run_backup_and_fingerprint_guard_on_real_files(tmp_path):
    import hashlib
    installer = Path(__file__).resolve().parents[1] / "ops/news_api/install_tls.py"
    source = ("def main():\n"
        "    def make_server(port):\n        server = ThreadingHTTPServer(('0.0.0.0', port), Handler)\n"
        "        return server\n    public = make_server(3001)\n"
        "    public.socket = tls.wrap_socket(public.socket, server_side=True)\n")
    target = tmp_path / "news_api.py"
    target.write_text(source, encoding="utf-8", newline="\n")
    original = target.read_bytes()
    command = [sys.executable, "-X", "utf8", str(installer), "--source", str(target),
               "--expected-sha256", hashlib.sha256(original).hexdigest()]
    subprocess.run(command, check=True, capture_output=True, timeout=10)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.bak.*"))
    subprocess.run([*command, "--apply"], check=True, capture_output=True, timeout=10)
    changed = target.read_bytes()
    assert b"BoundedTLSHTTPServer" in changed
    backups = list(tmp_path.glob("*.bak.*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert not list(tmp_path.glob(".tls-stage-*"))
    stale = subprocess.run([*command, "--apply"], capture_output=True, timeout=10)
    assert stale.returncode != 0 and target.read_bytes() == changed
