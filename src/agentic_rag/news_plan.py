"""校验规划器给出的新闻时间建议，不添加默认日期范围。"""

from datetime import date
import re
from typing import Any


def clean_queries(values: list[str], limit: int = 4) -> list[str]:
    """拆开模型误塞进单字段的查询列表，再限制数量并去重。"""
    queries = []
    for value in values:
        for part in re.split(r"[;；\n]+", value):
            query = " ".join(part.split())[:200]
            if query and query not in queries:
                queries.append(query)
    # 有明确主题时不混入“全部最新新闻”；只有全空时才保留不筛关键词。
    return queries[:limit] if queries else ([""] if values else [])


def prepare_news_plan(query: str, section: str, suggestion: dict[str, str]) -> dict[str, Any]:
    """保留模型原始建议，同时生成可以直接提交给 API 的条件。"""
    plan: dict[str, Any] = {
        "query": query.strip(),
        "section": section.strip(),
        "suggested_time": dict(suggestion),
        "start": "",
        "end": "",
        "time_note": "未设置时间条件",
        "error": "",
    }
    mode = suggestion.get("mode", "unrestricted")
    if mode == "unrestricted":
        if suggestion.get("start") or suggestion.get("end"):
            plan["time_note"] = "模型选择了不限时间，因此忽略附带的日期"
        return plan

    start = suggestion.get("start", "").strip()
    end = suggestion.get("end", "").strip()
    try:
        if mode not in {"suggested", "explicit"}:
            raise ValueError("时间类型无法识别")
        if not start and not end:
            raise ValueError("没有给出起止日期")
        if mode == "suggested" and not suggestion.get("reason", "").strip():
            raise ValueError("没有给出时间建议的理由")
        for value in (start, end):
            if value and date.fromisoformat(value).isoformat() != value:
                raise ValueError("日期必须使用 YYYY-MM-DD")
        if start and end and start > end:
            raise ValueError("开始日期晚于结束日期")
    except ValueError as exc:
        if mode == "explicit":
            plan["error"] = f"用户指定的新闻时间解析无效，停止本次新闻查询：{exc}"
            plan["time_note"] = plan["error"]
        else:
            plan["time_note"] = f"时间建议无效，按不限时间查询：{exc}"
        return plan

    plan.update(start=start, end=end)
    plan["time_note"] = "采用模型建议的时间范围" if mode == "suggested" else "采用用户指定的时间范围"
    return plan
