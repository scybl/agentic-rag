"""流程图节点与路由决策。

节点会修改状态；决策函数只读取状态，因此无需 LLM 即可对控制流程进行单元测试。
"""

import logging
import hashlib
import json
import re
import queue
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langgraph.config import get_stream_writer

from ..analysis_cache import AnalysisCache, build_cache_key, evidence_identity
from ..config import settings
from ..evidence import deduplicate, document_key, excerpt, format_evidence
from ..news_plan import clean_queries, prepare_news_plan
from ..tools import SOURCE_TOOLS
from ..tools.contracts import ToolContext
from ..tools.execution import execute_tool
from .chains import (
    get_document_grader,
    get_generator,
    get_answer_reviewer,
    get_evidence_assessor,
    get_router,
)
from .state import GraphState

logger = logging.getLogger(__name__)
ANALYSIS_PROMPT_VERSION = "research-v2-reading-specialists-reread"


def _select_answer_documents(documents: list[Document]) -> list[Document]:
    """按来源轮流选择证据，避免某一个来源占满模型上下文。"""
    limit = max(1, settings.generation_max_documents)
    groups: dict[str, list[Document]] = {}
    for document in documents:
        source_type = str(document.metadata.get("source_type") or "unknown")
        groups.setdefault(source_type, []).append(document)
    # 补搜后交替保留新旧轮次，避免新证据永远排在候选队列末尾。
    for source_type, group in groups.items():
        rounds = {}
        for document in group:
            rounds.setdefault(document.metadata.get("retrieval_round", 0), []).append(document)
        interleaved = []
        while any(rounds.values()):
            for round_number in sorted(rounds, reverse=True):
                if rounds[round_number]:
                    interleaved.append(rounds[round_number].pop(0))
        groups[source_type] = interleaved
    selected = []
    while len(selected) < limit and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < limit:
                selected.append(group.pop(0))
    return selected


def _format_docs(documents: list[Document], query: str = "") -> str:
    return format_evidence(documents, query, max(400, settings.generation_context_chars))


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------
def route(state: GraphState) -> GraphState:
    """规划本轮需要联合查询的来源，以及每个来源使用的查询。"""
    result = get_router().invoke({
        "question": state["question"],
        "original_question": state.get("original_question", state["question"]),
        "current_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
    })
    selected_sources = []
    source_queries = {}
    for enabled, source, query in (
        (result.use_knowledge, "vectorstore", result.knowledge_query),
        (result.use_news, "news_api", result.news_query),
        (result.use_web, "web_search", result.web_query),
    ):
        if enabled:
            selected_sources.append(source)
            source_queries[source] = query.strip() if source == "news_api" else (query.strip() or state["question"])
    if not selected_sources:
        selected_sources = ["web_search"]
        source_queries = {"web_search": state["question"]}
    logger.info("Planner -> %s", ", ".join(selected_sources))
    news_plan = prepare_news_plan(
        result.news_query, result.news_section, result.news_time.model_dump(),
    ) if result.use_news else {}
    if news_plan:
        news_plan["raw_queries"] = [result.news_query, *result.additional_news_queries]
        news_plan["queries"] = clean_queries(news_plan["raw_queries"])
        source_queries["news_api"] = " / ".join(news_plan["queries"])
    return {
        "datasource": "+".join(selected_sources),
        "selected_sources": selected_sources,
        "source_queries": source_queries,
        "plan_summary": result.plan_summary.strip(),
        "news_search_plan": news_plan,
        "task_type": result.task_type,
        "evidence_needs": result.evidence_needs,
        "original_question": state.get("original_question", state["question"]),
        "retries": state.get("retries", 0),
    }


def _source_writer(state: GraphState):
    """把工具事件送回主图线程，并为可恢复研究保留调用记录。"""
    if state.get("_event_writer") is not None:
        return state["_event_writer"]
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = lambda _event: None
    database = None
    if state.get("run_id") and state.get("reading_recipe"):
        from ..research.service import store
        database = store()

    def emit(event):
        if database is not None and event.get("kind") == "tool":
            database.event(state["run_id"], event)
        writer(event)
    return emit


def _search_source(source: str, arguments: dict, state: GraphState) -> GraphState:
    """节点只负责把工具证据写回状态，不再了解 HTTP、向量排序或转换细节。"""
    message = execute_tool(SOURCE_TOOLS[source], arguments,
                           context=ToolContext(emit=_source_writer(state)))
    bundle = message.artifact
    update = {"documents": bundle.documents, "datasource": source,
              "source_errors": {source: "；".join(bundle.warnings)} if bundle.warnings else {}}
    if source == "news_api":
        update["news_trace"] = bundle.events
    return update


def retrieve(state: GraphState) -> GraphState:
    return _search_source("vectorstore", {"query": state["question"]}, state)


def grade_documents(state: GraphState) -> GraphState:
    """只保留 LLM 评估器判定为相关的检索文本块。"""
    grader = get_document_grader()
    grades = dict(state.get("document_grades", {}))
    relevant = []
    question = state.get("original_question", state["question"])
    selection_query = question + " " + " ".join(state.get("evidence_needs", []))
    database = None
    if state.get("reading_recipe"):
        from ..research.service import store
        database = store()
    for doc in state["documents"]:
        key = document_key(doc)
        if key not in grades:
            text = f"标题：{doc.metadata.get('title', '')}\n" + excerpt(doc.page_content, selection_query, 2200)
            cache_key = hashlib.sha256(json.dumps(["grade-v2", question, text, state.get("model_revision")], ensure_ascii=False).encode()).hexdigest()
            cached = database.cache_get(cache_key) if database else None
            grades[key] = cached if cached is not None else grader.invoke({"document": text, "question": question}) == "yes"
            if database and cached is None:
                database.cache_put(cache_key, grades[key])
        if grades[key]:
            relevant.append(doc)
    logger.info("Document grading: %d/%d relevant", len(relevant), len(state["documents"]))
    grouped = {}
    for document in relevant:
        source_type = str(document.metadata.get("source_type") or "unknown")
        grouped.setdefault(source_type, []).append(document)
    return {"documents": relevant, "documents_by_source": grouped, "document_grades": grades}


def web_search(state: GraphState) -> GraphState:
    return _search_source("web_search", {"query": state["question"]}, state)


def news_api(state: GraphState) -> GraphState:
    """日期决策属于规划层；经校验后才把实际条件交给新闻工具。"""
    plan = state.get("news_search_plan") or prepare_news_plan(
        state["question"], "", {"mode": "unrestricted"},
    )
    if plan.get("error"):
        raise ValueError(plan["error"])
    return _search_source("news_api", {
        "semantic_query": state.get("original_question", state["question"]),
        "queries": plan.get("queries") or [plan["query"]],
        "start": plan["start"], "end": plan["end"], "section": plan["section"],
    }, state)


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    return deduplicate(documents)


def collect_sources(state: GraphState) -> GraphState:
    """独立来源受限并发；事件回到图线程输出，合并顺序保持稳定。"""
    documents_by_source = {}
    source_errors = {}
    all_documents = []
    news_trace = []
    history = list(state.get("query_history", []))
    handlers = {
        "vectorstore": retrieve,
        "news_api": news_api,
        "web_search": web_search,
    }
    events = queue.SimpleQueue()
    emit = _source_writer(state)

    def fetch(source):
        query = state["source_queries"].get(source, state["question"])
        try:
            return handlers[source]({
                "question": query,
                "original_question": state.get("original_question", state["question"]),
                "news_search_plan": state.get("news_search_plan", {}),
                "_event_writer": events.put,
            })
        except Exception as exc:
            logger.warning("Source %s failed: %s", source, exc, exc_info=True)
            return {"documents": [], "source_errors": {source: str(exc)}}
    pool = ThreadPoolExecutor(max_workers=max(1, min(settings.io_concurrency, 3)))
    fetched = {}
    try:
        active = {pool.submit(fetch, source): source for source in state["selected_sources"]}
        while active:
            done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
            while not events.empty():
                emit(events.get())
            for future in done:
                source = active.pop(future)
                fetched[source] = future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for source in state["selected_sources"]:
        query = state["source_queries"].get(source, state["question"])
        actual_queries = state.get("news_search_plan", {}).get("queries", [query]) if source == "news_api" else [query]
        history.extend(f"{source}: {value}" for value in actual_queries)
        result = fetched[source]
        documents = result.get("documents", [])
        source_errors.update(result.get("source_errors", {}))
        news_trace.extend(result.get("news_trace", []))
        for document in documents:
            document.metadata.setdefault("source_type", source)
        documents_by_source[source] = documents
        all_documents.extend(documents)
    live_ids = {d.metadata.get("article_id") for d in all_documents if d.metadata.get("article_id")}
    for document in state.get("memory_documents", []):
        if document.metadata.get("article_id") not in live_ids:
            all_documents.append(document)
            documents_by_source.setdefault("news_api", []).append(document)
    return {
        "documents_by_source": documents_by_source,
        "documents": _deduplicate_documents(all_documents),
        "source_errors": source_errors,
        "news_trace": news_trace,
        "query_history": history,
        "datasource": "+".join(state["selected_sources"]),
    }


SOURCE_NOTES = {
    "vectorstore": "the user's indexed knowledge files",
    "news_api": "the user's private news server",
    "web_search": "a public web search — NOT the user's documents",
}


def _source_note(documents: list[Document]) -> str:
    source_types = sorted(
        {str(document.metadata.get("source_type") or "unknown") for document in documents}
    )
    return "; ".join(SOURCE_NOTES.get(source, source) for source in source_types)


def _pending_queries(state: GraphState, review: dict) -> tuple[list[str], list[str]]:
    history = set(state.get("query_history", []))
    news = [q for q in clean_queries(review.get("news_queries", []), 3) if q and f"news_api: {q}" not in history]
    web = [q for q in clean_queries(review.get("web_queries", []), 2) if q and f"web_search: {q}" not in history]
    return news, web


def _can_supplement(state: GraphState) -> bool:
    research = bool(state.get("reading_recipe"))
    limit = settings.research_max_rounds if research else settings.max_retries
    if state.get("retries", 0) >= limit or state.get("no_progress_rounds", 0) >= 2:
        return False
    if research:
        from ..research.service import store
        if len(store().tasks(state["run_id"])) >= settings.research_max_tasks:
            return False
    return True


def assess_evidence(state: GraphState) -> GraphState:
    """先检查整批资料是否足以推断，再决定补搜或分析。"""
    documents = _select_answer_documents(state["documents"])
    question = state.get("original_question", state["question"])
    context = _format_docs(documents, question + " " + " ".join(state.get("evidence_needs", [])))
    review = get_evidence_assessor().invoke({
        "question": question, "task_type": state.get("task_type", "factual"),
        "current_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "evidence_needs": state.get("evidence_needs", []),
        "query_history": state.get("query_history", []), "documents": context,
    }).model_dump()
    valid_ids = {f"E{i}" for i in range(1, len(documents) + 1)}
    review["usable_evidence_ids"] = [eid for eid in review["usable_evidence_ids"] if eid in valid_ids]
    if not review["usable_evidence_ids"]:
        review["ready"] = False
    news, web = _pending_queries(state, review)
    can_search = _can_supplement(state) and bool(news or web)
    return {
        "evidence_assessment": review, "answer_documents": documents,
        "evidence_context": context,
        "pending_news_queries": news, "pending_web_queries": web,
        "next_action": "supplement" if not review["ready"] and can_search else "generate",
    }


def supplement_sources(state: GraphState) -> GraphState:
    """根据检查结果补查缺口，保留此前已取得的证据和日期选择。"""
    documents = list(state.get("documents", []))
    history = list(state.get("query_history", []))
    errors = {}
    trace = []
    news_queries = state.get("pending_news_queries", [])
    if news_queries:
        plan = dict(state.get("news_search_plan") or prepare_news_plan("", "", {"mode": "unrestricted"}))
        plan.update(query=news_queries[0], queries=news_queries)
        try:
            update = news_api({**state, "news_search_plan": plan})
            for doc in update["documents"]:
                doc.metadata["retrieval_round"] = state.get("retries", 0) + 1
            documents = update["documents"] + documents
            errors.update(update.get("source_errors", {}))
            trace.extend(update.get("news_trace", []))
        except Exception as exc:
            errors["news_api"] = str(exc)
        history.extend(f"news_api: {q}" for q in news_queries)
    for query in state.get("pending_web_queries", []):
        try:
            web_update = web_search({**state, "question": query})
            extra = web_update["documents"]
            errors.update(web_update.get("source_errors", {}))
            for doc in extra:
                doc.metadata["retrieval_round"] = state.get("retries", 0) + 1
            documents = extra + documents
        except Exception as exc:
            errors["web_search"] = str(exc)
        history.append(f"web_search: {query}")
    documents = _deduplicate_documents(documents)
    grouped = {}
    for doc in documents:
        grouped.setdefault(doc.metadata.get("source_type", "unknown"), []).append(doc)
    return {
        "documents": documents, "documents_by_source": grouped,
        "query_history": history, "source_errors": errors, "news_trace": trace,
        "retries": state.get("retries", 0) + 1,
    }


def _cache_version(context: str) -> str:
    # 证据顺序决定 E 编号，因此缓存还必须包含实际上下文，而非只有无序证据集合。
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    return f"{ANALYSIS_PROMPT_VERSION}:{today}:{digest}"


def _analysis_version(state, context):
    return _cache_version(context + json.dumps(state.get("specialist_findings", []), ensure_ascii=False)
                          + str(state.get("model_revision", "")))


def generate(state: GraphState) -> GraphState:
    question = state.get("original_question", state["question"])
    answer_documents = state.get("answer_documents", _select_answer_documents(state["documents"]))
    context = state.get("evidence_context") or _format_docs(answer_documents, question)
    cache_key, _, _ = build_cache_key(
        question,
        answer_documents,
        model=settings.llm_model,
        prompt_version=_analysis_version(state, context),
    )
    if settings.analysis_cache_enabled and state["documents"] and not state.get("revision_feedback"):
        cached = AnalysisCache(settings.analysis_cache_path).get(cache_key)
        if cached is not None and cached.strip():
            logger.info("Analysis cache hit")
            return {
                "generation": cached,
                "answer_documents": answer_documents,
                "analysis_cache_key": cache_key,
                "analysis_cache_hit": True,
                "evidence_context": context,
            }
    generation = get_generator().invoke(
        {
            "context": context,
            "question": question,
            "source_note": _source_note(answer_documents),
            "task_type": state.get("task_type", "factual"),
            "current_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
            "evidence_assessment": json.dumps(state.get("evidence_assessment", {}), ensure_ascii=False),
            "revision_feedback": state.get("revision_feedback", "首次回答"),
            # 原文回读已合并到有预算的证据上下文；不要在建议区重复塞入全部原文。
            "specialist_findings": json.dumps([{k: f.get(k) for k in ("goal", "summary", "evidence_ids", "limitations")}
                                                for f in state.get("specialist_findings", [])], ensure_ascii=False),
        }
    )
    return {
        "generation": generation,
        "answer_documents": answer_documents,
        "analysis_cache_key": cache_key,
        "analysis_cache_hit": False,
        "evidence_context": context,
    }


def revise_answer(state: GraphState) -> GraphState:
    """用反思反馈重写分析，引用同一份可核对证据。"""
    update = generate(state)
    update["answer_revisions"] = state.get("answer_revisions", 0) + 1
    return update


# --------------------------------------------------------------------------
# 路由决策（只读取状态的纯函数）
# --------------------------------------------------------------------------
def _is_unnecessary_forecast_refusal(state: GraphState) -> bool:
    """可分析却因缺少现成预测而拒答时，防止评审的误通过标记放行。"""
    if state.get("task_type") != "forecast" or not state.get("evidence_assessment", {}).get("ready"):
        return False
    answer = state.get("generation", "")
    refuses = re.search(r"(?:无法|不能|难以|不予).{0,16}(?:预测|分析|判断|给出)", answer)
    missing = re.search(r"(?:没有|未提供|缺乏|缺少|不包含|未包含).{0,45}(?:预测|未来|报告|资料|数据)", answer)
    # 正常条件预测可以同时声明无法精确预测，不能误伤这种不确定性说明。
    conditional_analysis = re.search(r"(?:基准情景|上行情景|下行情景|如果|若.{0,40}(?:则|可能|将)|有望|倾向于)", answer)
    return bool(refuses and missing and not conditional_analysis)


def evaluate_generation(state: GraphState) -> GraphState:
    """同时检查事实与任务完成情况，区分补证据和重写答案。"""
    answer_documents = state.get("answer_documents", state["documents"])
    context = state.get("evidence_context") or _format_docs(answer_documents, state["question"])
    if not state.get("generation", "").strip():
        review = {
            "decision": "revise",
            "grounded": False, "answers_question": False, "needs_more_evidence": False,
            "issues": ["模型没有返回可用答案"], "revision_instructions": "依据现有证据完成原始任务。",
            "news_queries": [], "web_queries": [],
        }
    else:
        review = get_answer_reviewer().invoke({
            "question": state.get("original_question", state["question"]),
            "task_type": state.get("task_type", "factual"),
            "evidence_assessment": json.dumps(state.get("evidence_assessment", {}), ensure_ascii=False),
            "documents": context, "generation": state["generation"],
        }).model_dump()
        # 结构化布尔值可能与文字结论冲突，不能仅凭两个 True 缓存答案。
        if review["decision"] != "accept" and review["grounded"] and review["answers_question"]:
            review["answers_question"] = False
            review["issues"].append("反思要求继续修正，答案尚未完成核验")
        instruction = review["revision_instructions"].strip().strip("。.!！ ").lower()
        has_corrections = bool(review["issues"]) or instruction not in {"", "无", "无需修改", "无需修正", "none"}
        if review["decision"] == "accept" and has_corrections:
            review.update(decision="revise", answers_question=False)
            review["issues"].append("反思文字提出了修正要求，与通过标记冲突，按需修正处理")
        if _is_unnecessary_forecast_refusal(state):
            review.update(decision="revise", answers_question=False, needs_more_evidence=False)
            review["issues"].append("已有可分析证据，却仅因缺少现成未来预测而拒答")
            review["revision_instructions"] = (
                "依据现有事实与明确假设，给出方向判断、基准/上行/下行情景和限制；不要等待现成未来答案。"
            )
        # 同时接受 [E1] 和 [E1, E2]，合并引用也要检查是否伪造编号。
        citation_groups = re.findall(r"\[(E\d+(?:\s*[,，、]\s*E\d+)*)\]", state["generation"])
        cited_ids = {eid for group in citation_groups for eid in re.findall(r"E\d+", group)}
        valid_ids = {f"E{i}" for i in range(1, len(answer_documents) + 1)}
        if cited_ids - valid_ids:
            review["grounded"] = False
            review["issues"].append("答案引用了不存在的证据编号：" + ", ".join(sorted(cited_ids - valid_ids)))
        if answer_documents and state.get("task_type") == "forecast" and not cited_ids:
            review["grounded"] = False
            review["issues"].append("预测中的事实需要使用 [E编号] 引用")
    passed = review["decision"] == "accept" and review["grounded"] and review["answers_question"]
    update = {
        "answer_review": review, "generation_grounded": review["grounded"],
        "generation_complete": review["answers_question"],
        "generation_check": "事实依据与任务完成情况均通过" if passed else "；".join(review["issues"]),
        "revision_feedback": "；".join([*review["issues"], review["revision_instructions"]]),
        "next_action": "finish",
    }
    if passed:
        cache_key = state.get("analysis_cache_key")
        if settings.analysis_cache_enabled and cache_key and answer_documents:
            evidence_hash, source_ids = evidence_identity(answer_documents)
            AnalysisCache(settings.analysis_cache_path).put(
                cache_key,
                question=state.get("original_question", state["question"]),
                answer=state["generation"],
                evidence_hash=evidence_hash,
                source_ids=source_ids,
                model=settings.llm_model,
                prompt_version=_analysis_version(state, context),
                ttl_seconds=settings.analysis_cache_ttl_seconds,
            )
        return update
    news, web = _pending_queries(state, review)
    if (review["needs_more_evidence"] or review["decision"] == "supplement") and (news or web) and _can_supplement(state):
        update.update(next_action="supplement", pending_news_queries=news, pending_web_queries=web)
    elif state.get("answer_revisions", 0) < settings.max_answer_revisions:
        update["next_action"] = "revise"
    else:
        update["generation"] = (
            "本次分析在补搜和修正预算内未通过最终核验。未将未通过的预测作为结论输出。\n"
            "仍需解决：" + "；".join(review["issues"])
        )
    return update


def decide_after_generation(state: GraphState) -> str:
    return state.get("next_action", "finish")
