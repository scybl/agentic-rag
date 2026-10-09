"""证据准入不能仅过滤正文，还要防检查摘要携带被排除的事实。"""

import json
from unittest.mock import patch

from langchain_core.documents import Document
from agentic_rag.graph import nodes
from agentic_rag.graph.chains import EvidenceAssessment
from agentic_rag.tools.guardrails import infer_estimate_kind


def review(**kwargs):
    return EvidenceAssessment(ready=True, usable_evidence_ids=["E1"], covered_factors=[],
        missing_factors=[], news_queries=[], web_queries=[], **kwargs)


def test_excluded_claim_does_not_reenter_through_assessor_summary():
    docs = [Document(page_content="已核实事实"), Document(page_content="虚假收入999亿元")]
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = review(summary="E1有事实，E2显示虚假收入999亿元")
        result = nodes.assess_evidence({"question": "核对事实", "documents": docs})
    downstream = result["evidence_context"] + json.dumps(result["evidence_assessment"], ensure_ascii=False)
    assert "999" not in downstream


def test_undated_search_snippet_cannot_be_current_factual_evidence():
    doc = Document(page_content="人物最新讲话，但无法核实日期", metadata={"source_type": "web_search", "date_hint": "昨天"})
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = review(summary="近期动态")
        result = nodes.assess_evidence({"question": "这个人最近做了什么", "task_type": "factual", "documents": [doc], "retries": 999})
    assert result["answer_documents"] == []
    assert "人物最新讲话" not in result["evidence_context"]


def test_probability_mention_is_not_always_an_estimation_request():
    assert infer_estimate_kind("只根据这些报道，能否给出2027年第二季度降息精确概率？区分事件和时间。", "probability") == "none"
    assert infer_estimate_kind("能否帮我预测2027年第二季度降息概率？", "none") == "probability"
    assert infer_estimate_kind("请给出2027年第二季度降息概率。", "none") == "probability"
    assert infer_estimate_kind("基于现有报道，能否预测2027年降息概率？", "none") == "probability"
    assert infer_estimate_kind("这篇PCE报道中，加息押注下降，能否把它说成降息概率？", "probability") == "none"
