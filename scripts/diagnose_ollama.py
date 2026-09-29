"""只读比较普通终端的代理设置与 Ollama 本机直连；不会输出密钥。"""

import sys
import time
import urllib.request
from urllib.parse import urlsplit

import httpx


def show_proxy(name, value):
    if name.lower() in {"no", "no_proxy"}:
        print("proxy bypass:", value, flush=True)
    else:
        parsed = urlsplit(value if "://" in value else "http://" + value)
        print(f"proxy/{name}: {parsed.hostname}:{parsed.port}", flush=True)


def main():
    from agentic_rag.config import settings

    from agentic_rag.ollama_connection import ollama_client_kwargs

    parsed = urlsplit(settings.ollama_base_url)
    if not ollama_client_kwargs(settings.ollama_base_url):
        raise SystemExit("This diagnostic only permits local Ollama endpoints")
    print("python:", sys.executable, flush=True)
    print(f"endpoint: {parsed.scheme}://{parsed.hostname}:{parsed.port}", flush=True)
    for name, value in urllib.request.getproxies().items():
        show_proxy(name, value)
    for trust_env in (True, False):
        started = time.perf_counter()
        try:
            with httpx.Client(trust_env=trust_env, timeout=10) as client:
                response = client.get(settings.ollama_base_url.rstrip("/") + "/api/version")
                print(
                    f"trust_env={trust_env} status={response.status_code} "
                    f"bytes={len(response.content)} server={response.headers.get('server', '')} "
                    f"seconds={time.perf_counter() - started:.2f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"trust_env={trust_env} error={type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()
