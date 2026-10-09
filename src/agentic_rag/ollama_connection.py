"""统一配置本机 Ollama 的连接方式，避免系统代理转发回环请求。"""

from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx


def ollama_client_kwargs(base_url: str) -> dict[str, bool]:
    """仅对本机地址关闭 HTTPX 代理继承；远程地址保留原有网络配置。"""
    url = base_url if "://" in base_url else f"http://{base_url}"
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    if hostname == "localhost":
        return {"trust_env": False}
    try:
        if ip_address(hostname).is_loopback:
            return {"trust_env": False}
    except ValueError:
        pass
    return {}


def safe_endpoint(base_url: str) -> str:
    """诊断仅展示连接目标，不泄露 URL 中的认证、路径或查询参数。"""
    try:
        url = urlsplit(base_url if "://" in base_url else f"http://{base_url}")
        host = url.hostname
        if not host:
            return "（地址格式无效，请检查 OLLAMA_BASE_URL）"
        host = f"[{host}]" if ":" in host else host
        return f"{url.scheme}://{host}" + (f":{url.port}" if url.port else "")
    except ValueError:
        return "（地址格式无效，请检查 OLLAMA_BASE_URL）"


def warmup_failure(exc: Exception, base_url: str) -> tuple[bool, str]:
    """预热错误策略：只重试明确的临时 HTTP 状态，不重复提交超时加载请求。"""
    endpoint = safe_endpoint(base_url)
    if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
        try:
            local = bool(ollama_client_kwargs(base_url))
        except ValueError:
            local = False
        action = (
            "请打开 Ollama 应用，或在另一个终端运行 ollama serve 并保持窗口开启，再重新启动项目。"
            if local else "请检查远程 Ollama 服务、网络和代理设置，再重新启动项目。"
        )
        return False, (
            f"无法建立到 Ollama（{endpoint}）的连接；尚未确认模型开始加载。"
            + action + "同时核对 OLLAMA_BASE_URL；本次不再重复预热。"
        )
    if isinstance(exc, httpx.TimeoutException):
        return False, (
            f"Ollama（{endpoint}）预热请求超时，不代表服务未启动；服务端可能仍在加载模型。"
            "请检查 Ollama 日志和内存/显存，等待后重试，必要时调整 LLM_REQUEST_TIMEOUT。"
            "为避免重复加载请求，本次不自动重试。"
        )
    status = getattr(exc, "status_code", None)
    if status in {429, 502, 503, 504}:
        return True, f"Ollama（{endpoint}）返回 HTTP {status}；可能暂时限流或不可用，不能仅据此判断正在加载模型。"
    if status == 404:
        return False, (
            f"Ollama（{endpoint}）返回 HTTP 404。请检查服务地址是否为 Ollama API，"
            "并运行 ollama list 核对 LLM_MODEL；如连接远程服务，须在对应服务器核对模型。"
            "本次不重试，也不会自动下载模型。"
        )
    if status in {401, 403}:
        return False, f"Ollama（{endpoint}）返回 HTTP {status}，请检查服务认证和访问权限；重复预热不能修复权限问题。"
    if status is not None:
        return False, f"Ollama（{endpoint}）返回 HTTP {status}；请检查服务日志、模型兼容性和资源，不盲目重复预热。"
    return False, f"预热发生 {type(exc).__name__}；请检查连接配置或客户端兼容性，本次不自动重试。"
