"""统一配置本机 Ollama 的连接方式，避免系统代理转发回环请求。"""

from ipaddress import ip_address
from urllib.parse import urlsplit


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
