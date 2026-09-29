"""抽取相关原文段落并去重，保留可核对证据，不生成新的事实。"""

import hashlib
import re

from langchain_core.documents import Document


def document_key(document: Document) -> str:
    return hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()


def excerpt(text: str, query: str, limit: int) -> str:
    """按问题选取原文片段；正文尾部的相关信息也有机会保留。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    words = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", query):
        words.update(run[index:index + 2] for index in range(len(run) - 1))
    parts = []
    for paragraph in re.split(r"(?<=[。！？；])|\n+", text):
        paragraph = paragraph.strip()
        parts.extend(paragraph[index:index + 260] for index in range(0, len(paragraph), 260))
    ranked = sorted(range(len(parts)), key=lambda i: (
        sum(word in parts[i].lower() for word in words)
        + 0.2 * bool(re.search(r"\d", parts[i])) + (0.1 if i == 0 else 0)
    ), reverse=True)
    selected: dict[int, str] = {}
    remaining = limit
    for index in ranked:
        if remaining < 40:
            break
        selected[index] = parts[index][:max(0, remaining - 2)]
        remaining -= len(selected[index]) + 2
    return "\n…".join(selected[index] for index in sorted(selected))[:limit]


def deduplicate(documents: list[Document]) -> list[Document]:
    seen_content, seen_articles, seen_titles = set(), set(), set()
    unique = []
    for document in documents:
        metadata = document.metadata
        content = re.sub(r"\s+", "", document.page_content)
        article_id = metadata.get("article_id")
        title = re.split(r"[_|]| - ", str(metadata.get("title") or ""))[0]
        title = re.sub(r"\s+|[….]", "", title)
        web_title = title if metadata.get("source_type") == "web_search" and len(title) >= 18 else ""
        if content in seen_content or (article_id and article_id in seen_articles) or (web_title and web_title in seen_titles):
            continue
        seen_content.add(content)
        if article_id:
            seen_articles.add(article_id)
        if web_title:
            seen_titles.add(web_title)
        unique.append(document)
    return unique


def format_evidence(documents: list[Document], query: str, budget: int) -> str:
    if not documents:
        return "（没有可用证据）"
    headers = []
    for index, document in enumerate(documents, 1):
        metadata = document.metadata
        title = str(metadata.get("title") or metadata.get("source") or "未知来源")[:100]
        kind = metadata.get("content_kind") or metadata.get("source_type", "unknown")
        headers.append(f"[E{index}] {title} | {metadata.get('published_at') or '日期未标注'} | {kind}\n")
    available = max(0, budget - sum(map(len, headers)) - 2 * (len(documents) - 1))
    per_document = available // len(documents)
    return "\n\n".join(
        header + excerpt(document.page_content, query, per_document)
        for header, document in zip(headers, documents)
    )[:budget]
