"""验证新闻时间只来自可见计划，不注入固定时间范围。"""

import pytest

from agentic_rag.news_plan import prepare_news_plan


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
