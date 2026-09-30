"""验证新闻时间只来自可见计划，不注入固定时间范围。"""

import pytest

from agentic_rag.news_plan import news_metadata_matches, prepare_news_plan


def test_default_has_no_date_filter():
    plan = prepare_news_plan("生猪", "", {})
    assert plan["start"] == plan["end"] == ""


def test_unrestricted_ignores_stray_dates_and_explains_it():
    suggestion = {"mode": "unrestricted", "start": "2026-09-29", "end": "2026-09-29"}
    plan = prepare_news_plan("生猪", "", suggestion)
    assert plan["start"] == plan["end"] == ""
    assert plan["suggested_time"] == suggestion
    assert "忽略" in plan["time_note"]


def test_valid_model_suggestion_is_used_without_a_fixed_window():
    plan = prepare_news_plan("生猪", "", {
        "mode": "suggested", "start": "2025-01-01", "end": "2026-09-29",
        "reason": "跨周期对比需要较长时间的资料",
    })
    assert (plan["start"], plan["end"]) == ("2025-01-01", "2026-09-29")
    assert "模型建议" in plan["time_note"]


@pytest.mark.parametrize("start,end,reason", [
    ("2026-02-30", "", "近期变化"),
    ("2026-10-01", "2026-09-01", "近期变化"),
    ("2026-09-01", "", ""),
])
def test_invalid_suggestion_is_visible_but_does_not_add_dates(start, end, reason):
    plan = prepare_news_plan("生猪", "", {
        "mode": "suggested", "start": start, "end": end, "reason": reason,
    })
    assert plan["start"] == plan["end"] == ""
    assert "时间建议无效" in plan["time_note"]


def test_relative_window_keeps_exact_timestamps_and_compiles_complete_instruction():
    plan = prepare_news_plan("马斯克 AI", "科技", {
        "mode": "explicit", "start": "", "end": "",
        "published_after": "2026-09-29T16:41:00+08:00",
        "published_before": "2026-09-30T16:41:00+08:00",
        "reason": "用户要求过去24小时",
    }, people=["马斯克"], organizations=["特斯拉"], topics=["人工智能"],
       source_names=["路透"], sort_by="newest", coverage="broad", result_limit=10)
    assert (plan["start"], plan["end"]) == ("2026-09-29", "2026-09-30")
    assert plan["published_after"].endswith("+08:00")
    assert plan["people"] == ["马斯克"] and plan["organizations"] == ["特斯拉"]
    assert plan["topics"] == ["人工智能"] and plan["source_names"] == ["路透"]
    assert plan["sort_by"] == "newest" and plan["coverage"] == "broad"
    assert plan["result_limit"] == 10 and "精确时间" in plan["time_note"]


def test_exact_time_and_source_are_hard_filters_for_live_and_memory_news():
    plan = prepare_news_plan("AI", "科技", {
        "mode": "explicit", "start": "", "end": "",
        "published_after": "2026-09-30T08:00:00+08:00",
        "published_before": "2026-09-30T10:00:00+08:00", "reason": "指定时段",
    }, source_names=["同花顺"])
    assert news_metadata_matches({
        "published_at": "2026-09-30T09:00:00+08:00", "section": "科技",
        "source_name": "tonghuashun", "url": "http://news.10jqka.com.cn/a",
    }, plan)
    assert not news_metadata_matches({
        "published_at": "2026-09-30T07:59:59+08:00", "section": "科技",
        "source_name": "tonghuashun",
    }, plan)
    assert not news_metadata_matches({
        "published_at": "2026-09-30T09:00:00+08:00", "section": "科技",
        "source_name": "other",
    }, plan)


def test_explicit_naive_timestamp_stops_query_instead_of_loosening_it():
    plan = prepare_news_plan("AI", "", {
        "mode": "explicit", "start": "", "end": "",
        "published_after": "2026-09-30T08:00:00", "published_before": "",
        "reason": "用户指定",
    })
    assert "停止本次新闻查询" in plan["error"]
