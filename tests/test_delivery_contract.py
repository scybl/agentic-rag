"""f6e38333 回归：最新50→影响20、预算饥饿、时间错配和伪完成。"""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from agentic_rag.config import settings
from agentic_rag.evidence import audit_numeric_claims, format_evidence
from agentic_rag.graph import nodes
from agentic_rag.impact_ranking import rank_impact
from agentic_rag.news_api import NewsClient, NewsAPIError
from agentic_rag.news_retrieval import retrieve_news
from agentic_rag.research import service
from agentic_rag.research.store import ResearchStore
from agentic_rag.research_contract import parse_contract, apply_contract, completion_violations, selection_appendix


QUESTION = "检索最近的与黄金相关的50篇新闻，找出影响最大的20篇，进行分析，预测2027年黄金价格走势"


@pytest.mark.parametrize("question", [QUESTION, QUESTION.replace("50", "五十").replace("20篇", "二十篇")])
def test_explicit_quantities_are_not_model_budgets(question):
    contract = parse_contract(question)
    assert contract == {"candidate_limit": 50, "selection_limit": 20, "latest": True, "ranking_mode": "impact"}
    plan = apply_contract({"query": "黄金", "queries": ["黄金", "美元指数"], "result_limit": 12, "start": "", "end": ""}, contract)
    assert plan["queries"] == ["黄金"] and plan["supporting_queries"] == ["美元指数"]
    assert plan["result_limit"] == 20 and plan["sort_by"] == "newest"
    assert plan["start"] == plan["end"] == ""
    assert parse_contract("预测2027年黄金价格走势") == {}


@pytest.mark.parametrize("question", ["从50篇新闻选60篇", "分析300篇新闻"])
def test_unsupported_counts_are_not_silently_clamped(question):
    with pytest.raises(ValueError):
        parse_contract(question)


class Client:
    iter_news = NewsClient.iter_news
    def __init__(self):
        self.calls, self.reads = [], []
        self.items = [{"article_id": str(i), "title": f"黄金消息{i}", "summary": "央行公布黄金储备变化。",
                       "published_at": f"2026-09-{30-i//24:02d}T{23-i%24:02d}:00:00Z"} for i in range(100)]
    def search(self, **kw):
        self.calls.append(kw)
        start = int(kw.get("cursor") or 0)
        end = start + kw["limit"]
        return {"items": self.items[start:end], "has_more": end < len(self.items), "next_cursor": str(end)}
    def article(self, article_id):
        self.reads.append(article_id)
        return {"article_id": article_id, "content": "正文事实。"}


def test_latest50_is_bounded_before_impact_and_vector_cache():
    client = Client()
    def rank(items, *_args):
        return [{**item, "candidate_id": item["article_id"]} for item in items], False, None
    def impact(items):
        return [{**item, "impact": {"score": int(item["article_id"]), "subject_relation": "direct"}} for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=rank):
        result = retrieve_news(semantic_query=QUESTION, api_query="黄金", candidate_limit=50,
                               result_k=20, sort_by="newest", ranking_mode="impact", client=client, scope_ranker=impact)
    assert len(client.calls) == 3 and len(client.reads) == 20
    assert [i["article_id"] for i in result.items] == [str(i) for i in range(49, 29, -1)]
    audit = next(e for e in result.events if e["kind"] == "news_ranked")
    assert audit["candidate_count"] == 50 and audit["selection_scope"] == "latest_n"
    assert audit["retrieval_complete"] and not audit["all_matches_exhausted"]


def test_latest_two_does_not_fetch_twenty_or_wait_for_five_to_score():
    class SparseClient(Client):
        def search(self, **kw):
            if kw["limit"] > 2:
                raise NewsAPIError("too much sparse-query scan")
            return super().search(**kw)
    client = SparseClient()
    scope = lambda items: [{**item, "impact": {"score": 10, "subject_relation": "direct"}} for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items,*args:(items,False,None)):
        result = retrieve_news(semantic_query="最新2篇黄金", api_query="黄金", candidate_limit=2,
            result_k=2, sort_by="newest", client=client, scope_ranker=scope)
    assert result.retrieval_complete and len(result.items) == 2
    assert len(client.calls) == 1 and client.calls[0]["limit"] == 2
    assert client.calls[0]["start"] == client.calls[0]["end"] == ""
    assert [x["article_id"] for x in result.items] == ["0", "1"]


def test_small_pages_continue_after_unrelated_news_instead_of_underdelivering():
    client = Client()
    def scope(items):
        return [{**item, "impact": {"score": 10,
            "subject_relation": "unrelated" if item["article_id"] == "0" else "direct"}} for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items,*args:(items,False,None)):
        result = retrieve_news(semantic_query="最新2篇黄金", api_query="黄金", candidate_limit=2,
            result_k=2, sort_by="newest", client=client, scope_ranker=scope)
    assert result.retrieval_complete
    assert [x["article_id"] for x in result.items] == ["1", "2"]
    assert len(client.calls) == 2 and all(x["limit"] == 2 for x in client.calls)


def test_scope_batch_rank_is_recomputed_for_the_whole_candidate_pool():
    client = Client()
    def scope(items):
        return [{**item, "impact": {"score": int(item["article_id"]), "rank": 1, "subject_relation": "direct"}}
                for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items,*args:(items,False,None)):
        result = retrieve_news(semantic_query="最新3篇黄金", api_query="黄金", candidate_limit=3,
            result_k=3, sort_by="newest", client=client, scope_ranker=scope)
    assert [x["article_id"] for x in result.items] == ["0", "1", "2"]  # 不改变用户要求的最新顺序。
    assert [x["impact"]["rank"] for x in result.items] == [3, 2, 1]  # 不是三个批内第一名。


def test_page_error_preserves_unscored_valid_candidates_without_claiming_complete():
    class PartialClient(Client):
        def __init__(self):
            super().__init__()
            self.items[0]["source_name"] = "target"
            self.items[1]["source_name"] = "other"
        def search(self, **kw):
            if kw.get("cursor"):
                raise NewsAPIError("next page timeout")
            return {"items": self.items[:2], "has_more": True, "next_cursor": "2"}
    client = PartialClient()
    scope = lambda items: [{**item, "impact": {"score": 10, "subject_relation": "direct"}} for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items,*args:(items,False,None)):
        result = retrieve_news(semantic_query="最新2篇黄金", api_query="黄金", candidate_limit=2,
            result_k=2, sort_by="newest", source_names=["target"], client=client, scope_ranker=scope)
    assert not result.retrieval_complete and "next page timeout" in result.api_error
    assert [x["article_id"] for x in result.items] == ["0"]


def test_impact_covers_all_candidates_and_reuses_exact_results(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    items = [{"candidate_id": str(i), "title": f"新闻{i}", "summary": "事实。"} for i in range(7)]
    calls = []
    def score(payload):
        batch = json.loads(payload["candidates"])
        calls.append(batch)
        return {"scores": [{"candidate_id": x["candidate_id"], "subject_relation": "direct", "magnitude": 3, "breadth": 2,
            "persistence": 2, "surprise": 1, "evidence": 2, "direction": "mixed", "mechanism": "可能影响供需", "support_ids": ["summary"]} for x in batch]}
    first = rank_impact(items, QUESTION, invoke=score, database=db)
    second = rank_impact(items, QUESTION, invoke=score, database=db)
    assert first == second and len(calls) == 2
    assert all(i["impact"]["score"] == 53.75 for i in first)


def test_impact_rejects_fabricated_support(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    with pytest.raises(ValueError, match="依据"):
        rank_impact([{"candidate_id": "a", "title": "新闻", "summary": ""}], QUESTION, database=db,
            invoke=lambda p: {"scores": [{"candidate_id": "N1", "subject_relation": "direct", "magnitude": 4, "breadth": 4, "persistence": 4,
                "surprise": 4, "evidence": 4, "direction": "positive", "mechanism": "冲击", "support_ids": ["summary"]}]})


class NoIndex:
    def flush(self): return {"synced": 0, "failed": 0, "remaining": 0}


def test_actual_chunks_get_budget_and_specialist_slots_reserved(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    docs = [Document(page_content="2026年1月，黄金新闻有事实。", metadata={"source_type": "news_api",
            "content_kind": "news_article", "article_id": str(i)}) for i in range(20)]
    state = {"documents": docs, "run_id": "r", "reading_recipe": "test", "model_revision": "v",
             "task_type": "forecast", "task_contract": parse_contract(QUESTION), "question": QUESTION}
    read = lambda p: {"claims": [{"kind": "reported_fact", "statement": "来源事实", "quote": p["text"]}], "limitations": []}
    result = service.read_documents(state, database=db, read_fn=read, index=NoIndex())
    assert result["task_budget"] == 23
    assert len(result["reading_reports"]) == 20 and all(r["status"] == "complete" for r in result["reading_reports"])
    specialist_state = {**state, **result, "answer_documents": result["documents"], "evidence_context": "[E1] 事实",
                        "evidence_needs": ["供给", "需求", "政策"]}
    findings = service.dispatch_specialists(specialist_state, database=db, index=NoIndex(), analyze_fn=lambda p: {
        "summary": "条件分析 [E1]", "evidence_ids": ["E1"], "limitations": [], "needs_more_evidence": False,
        "news_queries": [], "web_queries": []})
    assert findings["specialist_execution"] == {"expected": 3, "completed": 3}
    assert len(db.tasks("r")) == 23


def test_hard_budget_failure_cannot_pass_model_review(tmp_path):
    state = {"question": QUESTION, "documents": [], "task_contract": parse_contract(QUESTION),
             "reading_reports": [{"status": "partial", "covered": 0, "total": 1}],
             "specialist_execution": {"expected": 3, "completed": 0}, "generation": "模型宣称全部完成"}
    with patch.object(nodes, "get_answer_reviewer") as reviewer:
        result = nodes.evaluate_generation(state)
    assert not result["generation_complete"] and not reviewer.called
    assert result["next_action"] == "finish" and len(result["execution_violations"]) >= 3


def test_temporal_heading_survives_saved_reading_and_context(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    body = "第一阶段(1月)：金价冲顶\n国际黄金触及5595.75美元。\n情景一：展望2026年下半年\n预判金价4000至4800美元。"
    article = db.register(Document(page_content=body, metadata={"published_at": "2026-07-09T00:00:00Z"}), 2200)
    claims = [{"kind": "reported_fact", "statement": quote, "quote": quote} for quote in ["国际黄金触及5595.75美元。", "预判金价4000至4800美元。"]]
    saved = db.save_reading(article, "recipe", [(article["chunks"][0], {"claims": claims})])
    assert "1月" in saved["claims"][0]["event_context"]
    assert saved["claims"][1]["kind"] == "attributed_forecast"
    notes = [item["body"] for item in db.pending_vectors("test") if item["collection"] == "reading_memory"]
    assert notes and "attributed_forecast" in notes[0] and "第一阶段(1月)" in notes[0]
    doc = Document(page_content=body, metadata={"reading_claims": saved["claims"], "published_at": "2026-07-09"})
    context = format_evidence([doc], "黄金", 1000)
    assert "1月" in context and "attributed_forecast" in context
    assert audit_numeric_claims("7月金价触及5595.75美元 [E1]。", [doc], context)
    assert not audit_numeric_claims("1月金价触及5595.75美元 [E1]。", [doc], context)
    assert audit_numeric_claims("基准情景：2027年金价在4000–4800美元震荡。", [doc], context, forecast=True)


def test_twenty_news_are_not_squeezed_out_by_nine_document_limit():
    docs = [Document(page_content="原文", metadata={"source_type": "news_api", "article_id": str(i),
             "title": f"新闻{i}", "impact": {"rank": i+1}}) for i in range(20)]
    docs += [Document(page_content="方法", metadata={"source_type": "vectorstore"})]
    result = nodes._select_answer_documents(docs, parse_contract(QUESTION))
    assert len(result) == 21
    appendix = selection_appendix({"task_contract": parse_contract(QUESTION), "answer_documents": result})
    assert all(f"[E{i}]" in appendix for i in range(1,21))


def test_verbatim_short_article_stays_saved_but_is_not_dumped_into_appendix():
    original = "标题：产业动态\n正文：\nAI短剧内容合作仍在洽谈中，未披露签约金额。" + "展台接待客商，双方讨论交付安排。" * 45
    doc = Document(page_content=original, metadata={"source_type": "news_api", "article_id": "one",
        "title": "产业动态", "reading_claims": [{"kind": "source_excerpt", "statement": original}]})
    appendix = selection_appendix({"task_contract": {"selection_limit": 1}, "question": "AI短剧合作金额",
                                  "answer_documents": [doc]})
    assert "原文节选，未独立核实" in appendix
    assert "未披露签约金额" in appendix
    assert len(appendix) < 600 and original not in appendix
    assert doc.metadata["reading_claims"][0]["statement"] == original


def test_verified_batch_can_pass_but_missing_body_or_identity_cannot():
    docs = [Document(page_content="原文", metadata={"source_type": "news_api", "article_id": str(i),
             "content_kind": "news_article"}) for i in range(20)]
    state = {"task_contract": parse_contract(QUESTION), "task_type": "forecast", "answer_documents": docs,
        "reading_recipe": "test", "reading_reports": [{"status": "complete", "covered": 1, "total": 1}] * 20,
        "specialist_execution": {"expected": 3, "completed": 3},
        "news_trace": [{"kind": "news_ranked", "candidate_count": 50, "retrieval_complete": True,
                        "selection_scope": "latest_n", "subject_scope_checked": True, "ranking_method": "impact-v1"}]
                        + [{"kind": "news_selected", "article_id": str(i)} for i in range(20)]}
    assert completion_violations(state) == []
    state["specialist_execution"] = {"expected": 3, "completed": 0}
    assert completion_violations(state, final=False) == []
    assert any("专题分析" in message for message in completion_violations(state))
    state["specialist_execution"] = {"expected": 3, "completed": 3}
    docs[0].metadata["content_kind"] = "news_summary"
    docs[0].metadata["article_id"] = "outside-the-pool"
    violations = completion_violations(state)
    assert any("摘要" in message for message in violations)
    assert any("不一致" in message for message in violations)


def test_latest_window_counts_related_not_golden_week(tmp_path):
    client = Client()
    def scope(items):
        return [{**item, "impact": {"score": 70, "subject_relation": "unrelated" if int(item["article_id"]) < 25 else "direct"}}
                for item in items]
    with patch("agentic_rag.news_retrieval._rank_and_cache", side_effect=lambda items,*args:(items,False,None)):
        result = retrieve_news(semantic_query=QUESTION, api_query="黄金", candidate_limit=50, result_k=20,
                               sort_by="newest", ranking_mode="impact", client=client, scope_ranker=scope)
    assert len(client.calls) == 4 and {i["article_id"] for i in result.candidates} == {str(i) for i in range(25,75)}
    assert next(e for e in result.events if e["kind"] == "news_ranked")["unrelated_excluded"] == 25


def test_numbers_are_not_substring_matched():
    from agentic_rag.research.chains import numbers_supported
    assert not numbers_supported("涨幅2%", "涨幅12%")
    assert numbers_supported("涨幅12%", "涨幅12%")


def test_bounded_primary_query_keeps_explicit_entity():
    plan = apply_contract({"query": "黄金", "queries": ["黄金", "特朗普 黄金"], "people": ["特朗普"]}, parse_contract(QUESTION))
    assert plan["queries"] == ["黄金 特朗普"]


def test_empty_summary_support_gets_bounded_feedback(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    calls = []
    def invoke(payload):
        calls.append(payload)
        return {"scores": [{"candidate_id": "N1", "subject_relation": "direct", "magnitude": 2,
            "breadth": 2, "persistence": 2, "surprise": 2, "evidence": 4, "direction": "positive",
            "mechanism": "可能增加需求", "support_ids": ["summary"] if len(calls)==1 else ["title"]}]}
    result = rank_impact([{"candidate_id": "a", "title": "央行拟增持黄金", "summary": ""}], QUESTION, invoke=invoke, database=db)
    assert len(calls) == 2 and result[0]["impact"]["quote"] == "央行拟增持黄金"
    assert result[0]["impact"]["evidence"] == 1


def test_metadata_filter_cannot_bypass_latest_scan_limit():
    class FilteredClient:
        def iter_news(self, **kwargs):
            for i in range(2100):
                yield {"article_id": str(i), "title": "黄金", "source_name": "其他来源"}
    result = retrieve_news(semantic_query=QUESTION, api_query="黄金", candidate_limit=50,
        result_k=20, sort_by="newest", ranking_mode="impact", client=FilteredClient(),
        source_names=["指定来源"], scope_ranker=lambda items: items)
    assert not result.retrieval_complete and "2000" in result.api_error
    assert not result.items


def test_distinct_months_in_one_sentence_keep_their_own_prices():
    doc = Document(page_content="", metadata={"reading_claims": [
        {"quote": "1月金价5595.75美元", "event_context": "1月行情"},
        {"quote": "7月金价4500美元", "event_context": "7月行情"}]})
    context = "1月金价5595.75美元，7月金价4500美元"
    assert not audit_numeric_claims(context + " [E1]", [doc], context)
    assert audit_numeric_claims("7月金价5595.75美元 [E1]", [doc], context)


def test_separate_scenario_heading_and_price_range_are_not_a_loophole():
    answer = "#### 情景A：看跌\n* **目标区间：** 2500-2700美元/盎司 [E1]。"
    assert audit_numeric_claims(answer, [], "机构预测2500-2700美元", forecast=True)
    assert audit_numeric_claims("价格在2500–2700美元", [], "原文只有2700美元")
    assert audit_numeric_claims("目标区间：**2500–2700** 美元", [], "来源预测2500至2700美元", forecast=True)
    assert not audit_numeric_claims("历史报价4,300美元", [], "原文4300美元")
    assert not audit_numeric_claims("部分观点认为金价将到2500–2700美元；另一些投行预测5400–5600美元。", [],
                                   "机构预测2500 2700 5400 5600美元", forecast=True)


def test_internal_reference_cleanup_preserves_real_sources_and_links():
    from agentic_rag.evidence import normalize_answer_references
    assert normalize_answer_references("风险[C6]；事实[E2, C7]；[C6](https://example.com)") == "风险；事实[E2]；[C6](https://example.com)"


def test_appendix_does_not_turn_lower_hike_odds_into_rate_cuts():
    doc = Document(page_content="", metadata={"source_type": "news_api", "impact": {
        "mechanism": "降息预期升温推动金价上涨", "quote": "市场下调10月加息概率", "score": 60}})
    text = selection_appendix({"task_contract": parse_contract(QUESTION), "answer_documents": [doc]})
    assert "降息预期升温" not in text and "不代表降息预期" in text


def test_specialist_unverified_price_does_not_prime_final_generator():
    state = {"specialist_findings": [{"goal": "价格", "summary": "价格在4100–4300美元。利率压力需要跟踪 [E1]。",
                                     "evidence_ids": ["E1"], "limitations": []}]}
    checked = nodes._checked_specialist_advice(state, [], "证据只有4300美元")
    assert "4100" not in checked[0]["summary"] and "利率压力" in checked[0]["summary"]
    assert checked[0]["limitations"] and "4100" in state["specialist_findings"][0]["summary"]


def test_less_hiking_is_not_observed_rate_cutting():
    from agentic_rag.evidence import audit_policy_claims
    assert audit_policy_claims("降息预期升温，为金价提供了反弹动力。", "市场下调加息概率")
    assert not audit_policy_claims("若未来降息预期升温，金价可能受到支撑。", "市场下调加息概率")
    assert not audit_policy_claims("降息预期升温。", "市场直接报告降息概率上升")


def test_cli_unverified_answer_is_not_reported_as_execution_success():
    from agentic_rag.cli import ask
    class Graph:
        def stream(self, *args, **kwargs):
            yield "updates", {"generate": {"generation": "尚未通过", "generation_complete": False,
                                             "generation_grounded": False}}
    assert ask(Graph(), "测试") is False


def test_nested_tools_keep_child_identity():
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from agentic_rag.tools.contracts import ToolInput, ToolContext, EvidenceBundle, context_from
    from agentic_rag.tools.execution import execute_tool
    class EmptyInput(ToolInput):
        pass
    @tool(args_schema=EmptyInput, response_format="content_and_artifact")
    def outer(config: RunnableConfig):
        """测试嵌套事件。"""
        context_from(config).emit({"kind": "news_scope_excluded", "reason": "黄金周不是金价新闻"})
        context_from(config).emit({"kind": "tool", "tool": "child", "tool_call_id": "child-id",
                                   "caller": "子调用者", "reason": "子原因"})
        return EvidenceBundle().as_response()
    events = []
    execute_tool(outer, {}, context=ToolContext(emit=events.append, caller="外层", reason="外层原因"))
    child = next(e for e in events if e.get("tool") == "child")
    assert child["tool_call_id"] == "child-id" and child["caller"] == "子调用者"
    assert child["parent_tool_call_id"] == events[0]["tool_call_id"]
    excluded = next(e for e in events if e.get("kind") == "news_scope_excluded")
    assert excluded["reason"] == "黄金周不是金价新闻" and excluded["tool_reason"] == "外层原因"


def test_atomic_context_borrows_unused_space_without_cutting_a_fact():
    long_quote = "价格" * 140 + "必须连同条件保留。"
    long = Document(page_content="", metadata={"reading_claims": [{"kind": "reported_fact",
        "statement": "价格", "quote": long_quote, "event_context": "1月行情"}]})
    short = Document(page_content="短事实", metadata={})
    context = format_evidence([long, short], "价格", 600)
    assert long_quote in context and len(context) <= 600
    too_small = format_evidence([long, short], "价格", 150)
    assert "完整事实块超过上下文预算" in too_small and long_quote[:100] not in too_small


def test_specialist_reread_keeps_all_twenty_context_entries(tmp_path):
    db = ResearchStore(tmp_path / "research.sqlite")
    body = "黄金价格相关观测。" * 28
    article = db.register(Document(page_content=body), 2200)
    docs = [Document(page_content="笔记", metadata={"title": f"黄金{i}", "source_type": "news_api",
        "memory_version": article["version"], "reading_claims": [{"kind": "reported_fact", "statement": "价格",
        "quote": body, "event_context": "1月行情"}]}) for i in range(20)]
    state = {"run_id": "reread", "question": QUESTION, "task_type": "forecast", "model_revision": "test",
             "task_contract": parse_contract(QUESTION), "answer_documents": docs,
             "evidence_context": "已有证据", "evidence_needs": ["黄金价格"]}
    def analyze(payload):
        return {"summary": "条件判断 [E1]", "evidence_ids": ["E1"], "limitations": [],
            "needs_more_evidence": False, "news_queries": [], "web_queries": [],
            "reread_evidence_ids": [] if "原文回读结果" in payload["context"] else ["E1"]}
    config = replace(settings, llm_context_window=16384, llm_max_output_tokens=8192, generation_context_chars=6000)
    with patch.object(service, "settings", config):
        result = service.dispatch_specialists(state, database=db, index=NoIndex(), analyze_fn=analyze)
    context = result["evidence_context"]
    assert all(f"[E{i}]" in context for i in range(1, 21))
    assert "完整事实块超过上下文预算" not in context and len(context) <= 8884
