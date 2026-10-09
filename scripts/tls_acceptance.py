"""只读公网验收：验证证书、鉴权、只读约束和单个慢握手隔离；不输出凭据。"""

import argparse
import json
import socket
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request

from agentic_rag.news_api import NewsClient
from agentic_rag.telemetry.exporters import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--slow-handshake", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不能覆盖历史验收记录")
    client = NewsClient(base_url=args.base_url, timeout=4, max_attempts=1)
    endpoint = urlsplit(client.base_url)
    if endpoint.scheme != "https":
        raise ValueError("TLS 验收必须使用 HTTPS")
    checks = []
    result = {"started_at": time.time(), "transport": "direct public HTTPS; system CA verification enabled",
              "endpoint": client.base_url, "checks": checks}
    started = time.monotonic()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((endpoint.hostname, endpoint.port or 443), timeout=4) as raw:
            with context.wrap_socket(raw, server_hostname=endpoint.hostname) as conn:
                cert = conn.getpeercert()
                result["tls"] = {"version": conn.version(), "cipher": conn.cipher()[0],
                                 "not_before": cert.get("notBefore"), "not_after": cert.get("notAfter")}
        checks.append({"name": "verified_certificate", "passed": True})
        checks.append({"name": "authenticated_health", "passed": client.health().get("ok") is True})
        opener = client._opener  # 与真实客户端使用相同的禁代理、禁凭据重定向策略。
        for name, path, method, headers, expected in [
            ("unauthenticated_rejected", "/health", "GET", {}, 401),
            ("wrong_key_rejected", "/health", "GET", {"X-API-Key": "invalid-acceptance-key"}, 401),
            ("write_method_rejected", "/health", "POST", {"X-API-Key": client.api_key}, 405),
            ("invalid_cursor_rejected", "/v1/news?cursor=invalid", "GET", {"X-API-Key": client.api_key}, 400),
        ]:
            request = Request(client.base_url + path, headers=headers, method=method)
            try:
                with opener.open(request, timeout=4) as response:
                    status = response.status
            except HTTPError as exc:
                status = exc.code
                exc.close()
            checks.append({"name": name, "status": status, "passed": status == expected})
        if args.slow_handshake:
            # 只占一个连接，不发 TLS 字节；验收不是并发压测。
            with socket.create_connection((endpoint.hostname, endpoint.port or 443), timeout=11) as slow:
                held_at = time.monotonic()
                normal_at = time.monotonic()
                healthy = client.health().get("ok") is True
                elapsed = time.monotonic() - normal_at
                checks.append({"name": "healthy_during_stalled_handshake", "passed": healthy and elapsed < 4,
                               "seconds": elapsed})
                closed = slow.recv(1) == b""
                lifetime = time.monotonic() - held_at
                checks.append({"name": "stalled_handshake_reclaimed", "passed": closed and lifetime < 11,
                               "seconds": lifetime})
            checks.append({"name": "healthy_after_reclamation", "passed": client.health().get("ok") is True})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
    result["elapsed_seconds"] = time.monotonic() - started
    result["passed"] = not result.get("error") and len(checks) == (9 if args.slow_handshake else 6) and all(c["passed"] for c in checks)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
