"""真实验收前新增的确定性反例；不把模拟模型作为真实质量结果。"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from agentic_rag.budget import ResearchBudget, ResearchBudgetExceeded, research_budget, reserve_model_call
from agentic_rag.evaluation.answer_samples import actual_contexts
from agentic_rag.graph import nodes
from agentic_rag.graph.chains import EvidenceAssessment
from agentic_rag.research.chains import numbers_supported, resolve_selection


@pytest.mark.parametrize("summary,quote", [
    ("价格12元", "价格12万元"), ("占比3%", "提高3个百分点"),
    ("价格12美元", "价格12元"), ("涨幅2%", "涨幅12%"),
    ("费用5014.18万元", "费用-5014.18万元"),
    ("价格4215元/克", "价格4215美元/盎司"),
    ("规模6亿", "规模6万"),
])
def test_wrong_units_and_signs_rejected(summary, quote):
    assert not numbers_supported(summary, quote)


@pytest.mark.parametrize("summary,quote", [
    ("价格12万元", "价格12万元"), ("价格4215美元/盎司", "价格4,215美元/盎司"),
    ("同比3%", "同比3％"), ("提升3个百分点", "提升3个百分点"),
])
def test_exact_units_and_numeric_formatting(summary, quote):
    assert numbers_supported(summary, quote)


def test_bad_summary_reverts_to_quote_not_extra_model_retry():
    result = resolve_selection({"claims": [{"kind": "reported_fact", "statement": "价格12元",
        "sentence_ids": ["S1"]}], "limitations": []}, "价格12万元", {"S1": (0, 6)})
    assert result["claims"][0]["statement"] == "价格12万元"


def test_admission_filters_and_remaps_every_downstream_evidence_id():
    docs = [Document(page_content=text) for text in ["未确认旧闻", "可用的最新报道", "另一个不相干来源"]]
    review = EvidenceAssessment(ready=True, usable_evidence_ids=["E2", "E99"], covered_factors=["依据E2"],
        missing_factors=[], news_queries=[], web_queries=[], summary="采用E2", concerns=[
        {"category": "time", "evidence_ids": ["E2"], "detail": "E2需核对时间"}])
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = review
        result = nodes.assess_evidence({"question": "新闻", "documents": docs, "retries": 999})
    assert result["answer_documents"] == [docs[1]]
    assert "未确认旧闻" not in result["evidence_context"]
    assert "[E1]" in result["evidence_context"] and "[E2]" not in result["evidence_context"]
    assert result["evidence_assessment"]["usable_evidence_ids"] == ["E1"]
    assert result["evidence_assessment"]["concerns"][0]["evidence_ids"] == ["E1"]


def test_model_accept_cannot_override_obvious_arithmetic_conflict():
    from agentic_rag.graph.chains import AnswerAssessment
    doc = Document(page_content="用户规模预计突破6亿，较上年的1.2亿实现翻倍增长。")
    with patch.object(nodes, "get_answer_reviewer") as reviewer:
        reviewer.return_value.invoke.return_value = AnswerAssessment(decision="accept", grounded=True,
            answers_question=True, needs_more_evidence=False, issues=[], revision_instructions="",
            news_queries=[], web_queries=[], concern_checks=[])
        result = nodes.evaluate_generation({"question": "行业发展怎样", "task_type": "analysis",
            "documents": [doc], "answer_documents": [doc], "evidence_context": "[E1]" + doc.page_content,
            "generation": "用户规模预计突破6亿，较上年的1.2亿实现翻倍增长[E1]。"})
    assert not result["generation_grounded"] and not result["generation_complete"]
    assert "不能直接称为翻倍" in result["generation_check"]


def test_ragas_does_not_read_unseen_fulltext():
    assert actual_contexts({"evidence_context": "片段", "documents": [Document(page_content="未见全文")]}) == ["片段"]
    assert actual_contexts({"evidence_context": ""}) == []
    with pytest.raises(ValueError):
        actual_contexts({"documents": []})


def test_model_accept_does_not_merge_quarterly_gdp_into_monthly_pce():
    from agentic_rag.graph.chains import AnswerAssessment
    doc = Document(page_content="美国8月核心PCE环比增长0.2%。美国第二季度实际GDP年化季率终值为2.2%。")
    with patch.object(nodes, "get_answer_reviewer") as reviewer:
        reviewer.return_value.invoke.return_value = AnswerAssessment(decision="accept", grounded=True,
            answers_question=True, needs_more_evidence=False, issues=[], revision_instructions="",
            news_queries=[], web_queries=[], concern_checks=[])
        result = nodes.evaluate_generation({"question": "这些报道涉及什么时期", "task_type": "fact",
            "documents": [doc], "answer_documents": [doc], "evidence_context": "[E1]" + doc.page_content,
            "generation": "内容仅涉及2026年8月的经济数据（如PCE、GDP）[E1]。"})
    assert not result["generation_complete"]
    assert "统计期" in result["generation_check"]


@pytest.mark.parametrize("answer,blocked", [
    ("2026年8月的经济数据（如PCE、GDP）[E1]。", True),
    ("8月GDP同比增长2.2%[E1]。", True),
    ("第二季度GDP年化季率2.2%，8月PCE环比0.2%[E1]。", False),
    ("8月公布的第二季度GDP数据[E1]。", False),
    ("不能把第二季度GDP误称为8月GDP数据[E1]。", False),
])
def test_indicator_statistical_period_is_not_publication_date(answer, blocked):
    from agentic_rag.evidence import audit_indicator_periods
    context = "美国8月核心PCE环比增长0.2%。美国第二季度实际GDP年化季率终值为2.2%。"
    assert bool(audit_indicator_periods(answer, context)) is blocked


def test_monthly_gdp_in_source_is_not_overridden_by_a_quarter_elsewhere():
    from agentic_rag.evidence import audit_indicator_periods
    assert not audit_indicator_periods("8月GDP数据为1。", "甲国二季度GDP数据为2。乙国8月GDP数据为1。")


def test_factual_assessment_has_a_short_budget_without_disabling_analysis_thinking(monkeypatch):
    from dataclasses import replace
    from agentic_rag.graph import chains
    monkeypatch.setattr(chains, "settings", replace(chains.settings, llm_reasoning=True, llm_max_output_tokens=8192))
    fact = chains._initial_attempt_state("FactualEvidenceAssessment")
    assert fact.num_predict == 1400 and not fact.reasoning
    assert chains._initial_attempt_state("EvidenceAssessment").reasoning
    assert chains._initial_attempt_state("ResearchAnswer").reasoning


def test_call_budget_is_atomic_and_includes_failures():
    budget = ResearchBudget(calls=3)
    def attempt(_):
        try:
            budget.model_call()
            return True
        except ResearchBudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(30))) == 3
    assert budget.calls == 3


def test_resume_budget_uses_active_seconds_and_previous_calls():
    now = [0]
    budget = ResearchBudget(seconds=10, calls=3, elapsed=7, used_calls=2, clock=lambda: now[0])
    budget.model_call()
    with pytest.raises(ResearchBudgetExceeded):
        budget.model_call()
    now[0] = 4
    with pytest.raises(ResearchBudgetExceeded):
        budget.check()
    with research_budget(calls=1):
        reserve_model_call()
        with pytest.raises(ResearchBudgetExceeded):
            reserve_model_call()


def test_exhausted_run_does_not_permanently_poison_shared_reading_task(tmp_path):
    from agentic_rag.research.scheduler import Scheduler, TaskSpec
    from agentic_rag.research.store import ResearchStore
    db = ResearchStore(str(tmp_path / "budget.db"))
    db.start_run("limited", "预算不足")
    spec = TaskSpec("shared-reading", "reader", {})
    def handle(_):
        reserve_model_call()
        return {"ok": True}
    with research_budget(calls=1):
        reserve_model_call()
        with pytest.raises(ResearchBudgetExceeded):
            Scheduler(db, "limited").run([spec], {"reader": handle})
    assert db.task(spec.key)["status"] == "pending"
    assert db.task(spec.key)["retryable"]
    db.start_run("resumed", "提高预算后继续")
    with research_budget(calls=2):
        result = Scheduler(db, "resumed").run([spec], {"reader": handle})
    assert result[spec.key] == {"ok": True}


def test_actual_structured_attempts_cannot_bypass_shared_budget():
    from agentic_rag.graph import chains
    from langchain_core.messages import AIMessage
    from pydantic import BaseModel
    class Output(BaseModel):
        value: str
    class Model:
        def __init__(self):
            self.calls = 0
        def with_structured_output(self, *_args, **_kwargs):
            return self
        def invoke(self, *_args, **_kwargs):
            self.calls += 1
            return {"raw": AIMessage(content='{}'), "parsed": Output(value="done"), "parsing_error": None}
    model = Model()
    with patch.object(chains, "get_llm", return_value=model):
        chain = chains.checked_structured(Output)
        with research_budget(calls=1):
            assert chain.invoke("first").value == "done"
            with pytest.raises(ResearchBudgetExceeded):
                chain.invoke("second")
    assert model.calls == 1


def test_cli_budget_stop_preserves_failure_not_success(capsys):
    from agentic_rag import cli
    from dataclasses import replace
    class Graph:
        def stream(self, *args, **kwargs):
            reserve_model_call()
            reserve_model_call()
            yield "updates", {"generate": {"generation": "不应输出", "generation_complete": True, "generation_grounded": True}}
    with patch.object(cli, "settings", replace(cli.settings, research_max_model_calls=1)):
        assert not cli.ask(Graph(), "预算反例")
    output = capsys.readouterr().out
    assert "达到预算" in output and "不应输出" not in output


def test_resume_display_does_not_consume_execution_budget(capsys):
    from agentic_rag import cli
    from types import SimpleNamespace
    from dataclasses import replace
    # 未完成恢复需要版本校验；已完成离线入口由 CLI.main 独立处理，避免触发模型。
    result = cli.run_with_trace(type("Graph", (), {"get_state": lambda self, _: SimpleNamespace(
        values={"generation": "历史结果", "question": "历史问题"}, next=(), tasks=())})(), "", run_id="saved", resume=True)
    assert result["generation"] == "历史结果"


def test_acceptance_retains_failed_rows_in_denominator(tmp_path):
    from agentic_rag.evaluation.acceptance import summarize
    from agentic_rag.telemetry.exporters import write_json
    write_json(tmp_path / "N01-basic.json", {"case_id": "N01", "variant": "basic",
        "error": {"type": "timeout"}, "elapsed_seconds": 3, "workflow_passed": False,
        "checks": {"checks_passed": False}, "usage": None})
    report = summarize(tmp_path, 2)
    assert report["missing"] == 1
    assert report["variants"]["basic"]["count"] == 1
    assert report["variants"]["basic"]["errors"] == 1
    assert report["variants"]["basic"]["joint_passed"] == 0
    assert report["variants"]["basic"]["unknown_usage_records"] == 1


def test_frozen_source_strips_prior_model_opinions(tmp_path, monkeypatch):
    import json
    import sqlite3
    from agentic_rag.evaluation import acceptance
    from agentic_rag.telemetry.exporters import write_json
    source = tmp_path / "archive.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE versions(id, body, metadata)")
        db.execute("INSERT INTO versions VALUES(?,?,?)", ("abc123", "原文价格12万元", json.dumps({
            "title": "真实标题", "impact": {"mechanism": "先前模型猜测"}, "relevance_reason": "旧评分",
            "source_type": "news_api", "source": "https://example.org/article"})))
    cases = tmp_path / "cases.json"
    write_json(cases, {"cases": [{"id": "N1", "sources": ["abc"], "quotes": ["12万元"], "required": [], "forbidden": []}]})
    monkeypatch.setattr(acceptance, "CASES", cases)
    acceptance.freeze(tmp_path / "snapshot", database=source)
    saved = json.loads((tmp_path / "snapshot/snapshot.json").read_text(encoding="utf-8"))
    doc = saved["documents"]["abc"]
    assert doc["page_content"] == "原文价格12万元"
    assert "impact" not in doc["metadata"] and "relevance_reason" not in doc["metadata"]
