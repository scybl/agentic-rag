"""同花顺新闻 API 的只读客户端。

本模块不依赖 RAG 流程图或本地数据库。请将 API 密钥保存在 NEWS_API_KEY 中
（或显式传入），切勿将其放进 URL。
"""

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .news_plan import validate_api_query

try:
    from dotenv import load_dotenv
    from .config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - the client also works without python-dotenv
    pass

DEFAULT_BASE_URL = "https://106.54.27.114:3001"


class NewsAPIError(RuntimeError):
    """请求失败或服务器返回了非预期响应。"""

    def __init__(self, message: str, status_code: int | None = None, retry_after: float | None = None,
                 *, retryable: bool = True, error_code: str = "", request_id: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable
        self.error_code = error_code
        self.request_id = request_id


def _request_id(value) -> str:
    """仅接受服务生成的关联编号，避免任意响应内容进入错误日志。"""
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{32}", value) else ""


class _NoCredentialRedirect(HTTPRedirectHandler):
    """API 地址须显式配置；禁止重定向绕过 HTTPS/目标主机约束并携带密钥。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class NewsClient:
    """从服务器获取新闻列表、文章全文和统计信息。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15,
        max_attempts: int = 3,
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("NEWS_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        parsed = urlsplit(self.base_url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("NEWS_API_BASE_URL must be a server URL without credentials or a query")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("Remote news API connections must use HTTPS")

        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("Set NEWS_API_KEY or pass api_key to NewsClient")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be an integer from 1 to 3")
        self.max_attempts = max_attempts
        self.on_retry = on_retry
        # 不将密钥转发给从环境中继承的代理服务器。
        self._opener = build_opener(ProxyHandler({}), _NoCredentialRedirect())

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .research.runtime import io_capacity, task_context
        for attempt in range(1, self.max_attempts + 1):
            from .budget import check_budget
            check_budget()
            try:
                with io_capacity().slot():
                    return self._request(path, params)
            except NewsAPIError as exc:
                delay = exc.retry_after if exc.retry_after is not None else 0.5 * attempt
                if (not exc.retryable or exc.status_code not in {429, 502, 503, 504}
                        or attempt == self.max_attempts or not 0 <= delay <= 5):
                    raise
                if self.on_retry:
                    self.on_retry({"kind": "news_retry", "status_code": exc.status_code,
                                   "attempt": attempt + 1, "max_attempts": self.max_attempts,
                                   "delay": delay, "query": (params or {}).get("q", ""),
                                   "error_code": exc.error_code, "request_id": exc.request_id,
                                   "path": path})
                # 等待时不占 I/O 槽位；有任务上下文时支持取消，避免超时后继续提交请求。
                context = task_context.get()
                if context:
                    if context["cancel"].wait(delay) or time.monotonic() >= context["deadline"]:
                        raise TimeoutError("新闻重试已取消或超时") from exc
                else:
                    time.sleep(delay)

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"X-API-Key": self.api_key}, method="GET")
        from .budget import active_budget
        budget = active_budget.get()
        if budget:
            budget.check()
        remaining = budget.remaining() if budget else None
        timeout = min(self.timeout, max(0.1, remaining)) if remaining is not None else self.timeout
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = json.load(response)
                request_id = _request_id((getattr(response, "headers", None) or {}).get("X-Request-ID"))
        except HTTPError as exc:
            # 服务用 503 同时表示临时不可用和查询超时；只识别已知错误协议，
            # 不把任意响应正文（可能含代理页面/敏感信息）回显到终端或模型。
            try:
                error_payload = json.loads(exc.read(4096))
            except (ValueError, OSError):
                error_payload = {}
            if not isinstance(error_payload, dict):
                error_payload = {}
            message = error_payload.get("error", "")
            known_codes = {"query_timeout", "database_unavailable", "busy", "internal_error"}
            error_code = error_payload.get("error_code")
            error_code = error_code if isinstance(error_code, str) and error_code in known_codes else ""
            request_id = _request_id((exc.headers or {}).get("X-Request-ID")) or _request_id(error_payload.get("request_id"))
            legacy_deadline = (exc.code == 503 and not error_code and isinstance(message, str)
                              and message.startswith("Query unavailable or exceeded ")
                              and message.endswith(" seconds; narrow the date range"))
            query_deadline = error_code == "query_timeout" or legacy_deadline
            retry_after = None
            hint = (exc.headers or {}).get("Retry-After")
            if hint:
                try:
                    retry_after = float(hint)
                except ValueError:
                    # HTTP 日期格式或不可解析提示不猜测等待时间，交由用户稍后重试。
                    retry_after = float("inf")
            exc.close()
            description = f"News API returned HTTP {exc.code}"
            if 300 <= exc.code < 400:
                description += "：拒绝自动跳转，防止认证密钥被转发；请核对 NEWS_API_BASE_URL"
            elif query_deadline:
                description += "：服务端报告查询不可用或超过查询时限；停止重试相同请求，未自动缩小时间范围"
            elif error_code == "database_unavailable":
                description += "：新闻数据库暂时不可用"
            elif error_code == "busy":
                description += "：新闻接口并发槽位已满"
            elif error_code == "internal_error":
                description += "：新闻服务内部错误，不应当作查询超时"
            if request_id:
                description += f" [request_id={request_id}]"
            retryable = (exc.code in {429, 502, 503, 504} and not query_deadline
                         and error_code != "internal_error" and error_payload.get("retryable") is not False)
            raise NewsAPIError(description, exc.code, retry_after, retryable=retryable,
                               error_code=error_code, request_id=request_id) from exc
        except URLError as exc:
            raise NewsAPIError(f"Could not connect to the news API ({type(exc.reason).__name__})") from exc
        except TimeoutError as exc:
            raise NewsAPIError("News API request timed out") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NewsAPIError("News API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise NewsAPIError("News API returned an unexpected response")
        if request_id:
            payload["request_id"] = request_id
        return payload

    def health(self) -> dict[str, Any]:
        """检查服务和数据库健康状态；此端点同样需要密钥。"""
        return self._get("/health")

    def search(
        self,
        q: str = "",
        start: str = "",
        end: str = "",
        section: str = "",
        limit: int = 20,
        cursor: str = "",
        order: str = "desc",
        include_content: bool = False,
    ) -> dict[str, Any]:
        """获取一页新闻；日期是包含边界的 Asia/Shanghai 日期。"""
        validate_api_query(q, section)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if order not in {"asc", "desc"}:
            raise ValueError("order must be 'asc' or 'desc'")
        params: dict[str, Any] = {"limit": limit, "order": order}
        for name, value in (
            ("q", q), ("start", start), ("end", end), ("section", section), ("cursor", cursor)
        ):
            if value:
                params[name] = value
        if include_content:
            params["include_content"] = "true"
        return self._get("/v1/news", params)

    def article(self, article_id: str) -> dict[str, Any]:
        """根据文章 ID 获取一篇文章的全文。"""
        if not article_id:
            raise ValueError("article_id must not be empty")
        payload = self._get("/v1/news/" + quote(str(article_id), safe=""))
        item = payload.get("item")
        if not isinstance(item, dict):
            raise NewsAPIError("News API returned an article without an item")
        return item

    def stats(self) -> dict[str, Any]:
        """获取文章总数和栏目统计信息。"""
        return self._get("/v1/stats")

    def iter_news(
        self,
        q: str = "",
        start: str = "",
        end: str = "",
        section: str = "",
        page_size: int = 100,
        max_items: int | None = 100,
        order: str = "desc",
        include_content: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """跨页逐条生成新闻；传入 max_items=None 可获取所有匹配结果。"""
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer from 1 to 100")
        if max_items is not None and (
            isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0
        ):
            raise ValueError("max_items must be a non-negative integer or None")
        cursor = ""
        seen_cursors: set[str] = set()
        count = 0
        page_number = 0
        while max_items is None or count < max_items:
            page_number += 1
            limit = page_size if max_items is None else min(page_size, max_items - count)
            if on_event is not None:
                on_event({
                    "kind": "news_request", "page": page_number,
                    "query": q, "start": start, "end": end, "section": section,
                    "limit": limit, "continuation": bool(cursor), "order": order,
                    "max_items": max_items,
                })
            page = self.search(
                q=q,
                start=start,
                end=end,
                section=section,
                limit=limit,
                cursor=cursor,
                order=order,
                include_content=include_content,
            )
            items = page.get("items")
            if not isinstance(items, list):
                raise NewsAPIError("News API returned a page without items")
            if on_event is not None:
                on_event({
                    "kind": "news_page", "page": page_number,
                    "count": len(items), "has_more": bool(page.get("has_more")),
                    "request_id": _request_id(page.get("request_id")),
                })
            for item in items:
                if not isinstance(item, dict):
                    raise NewsAPIError("News API returned an invalid news item")
                yield item
                count += 1
                if max_items is not None and count >= max_items:
                    return
            if not page.get("has_more"):
                return
            if not items:
                raise NewsAPIError("News API returned an empty page with has_more=true")
            next_cursor = page.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                raise NewsAPIError("News API returned a missing or repeated pagination cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
