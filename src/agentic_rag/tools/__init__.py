"""显式工具目录；导入不会连接模型、访问网络或建立索引。

仅把模型可请求的取证能力暴露为工具，调度、缓存写入和反思不注册。
"""

from .knowledge import search_knowledge
from .news import search_news
from .news_reading import read_news
from .web_search import search_web


SOURCE_TOOLS = {
    "vectorstore": search_knowledge,
    "news_api": search_news,
    "web_search": search_web,
}
TOOLS = (*SOURCE_TOOLS.values(), read_news)


def describe_tools() -> str:
    """从实际工具定义生成规划提示，避免工具名字与实现各写一套。"""
    descriptions = []
    for source, item in SOURCE_TOOLS.items():
        inputs = ", ".join(item.tool_call_schema.model_json_schema()["properties"])
        descriptions.append(f"{source} -> {item.name}({inputs}): {item.description}")
    descriptions.append(f"专题回读 -> {read_news.name}(evidence_ids, goal): {read_news.description}")
    return "\n".join(descriptions)


__all__ = ["TOOLS", "SOURCE_TOOLS", "describe_tools", "search_knowledge", "search_news", "search_web", "read_news"]
