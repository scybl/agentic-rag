"""同花顺新闻 API 的只读客户端。

本模块不依赖 RAG 流程图或本地数据库。请将 API 密钥保存在 NEWS_API_KEY 中
（或显式传入），切勿将其放进 URL。
"""

import json
import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - the client also works without python-dotenv
    pass

DEFAULT_BASE_URL = "https://106.54.27.114:3001"


class NewsAPIError(RuntimeError):
    """请求失败或服务器返回了非预期响应。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class NewsClient:
    """从服务器获取新闻列表、文章全文和统计信息。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15,
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
        # 不将密钥转发给从环境中继承的代理服务器。
        self._opener = build_opener(ProxyHandler({}))

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        from .research.runtime import io_capacity
        with io_capacity().slot():
            return self._request(path, params)

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"X-API-Key": self.api_key}, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise NewsAPIError(f"News API returned HTTP {exc.code}", exc.code) from exc
        except URLError as exc:
            raise NewsAPIError("Could not connect to the news API") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NewsAPIError("News API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise NewsAPIError("News API returned an unexpected response")
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
