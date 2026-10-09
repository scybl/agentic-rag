"""把持久化任务、阅读记忆和专题 Agent 接入主流程。"""

import json
import re
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from ollama import Client

from ..config import settings
from ..ollama_connection import ollama_client_kwargs
from .chains import specialist, validate_reading, read_segment
from .memory import MemoryIndex
from .scheduler import Scheduler, TaskSpec, ResearchWorkPending
from .store import ResearchStore, fingerprint
from .runtime import task_context
from ..evidence import format_evidence
from ..tools import read_news
from ..tools.contracts import ToolContext
from ..tools.execution import execute_tool


READING_VERSION = "reader-v10-verbatim-source"
SPECIALIST_VERSION = "specialist-v10-current-date"
WORKFLOW_VERSION = "research-workflow-v22-abstract-first"


def store():
    return ResearchStore(settings.research_db)


def writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda event: None


def model_revision():
    # 每次新研究读取模型摘要，避免同名模型换权重后继续命中过期阅读成果。
    client = Client(host=settings.ollama_base_url,
                    **{**ollama_client_kwargs(settings.ollama_base_url), "timeout": 10})
    models = client.list().models
    name = settings.llm_model
    for model in models:
        if model.model in {name, name + ":latest"}:
            return fingerprint([name, model.digest, settings.temperature, settings.llm_context_window,
                                settings.llm_max_output_tokens, settings.llm_reasoning])
    raise RuntimeError("无法确认本地模型版本，停止建立可能失效的阅读缓存")


def recipe(signature):
    return fingerprint([READING_VERSION, signature, settings.reading_chunk_chars, settings.reading_verbatim_max_chars])


def initialize(state):
    run_id = state.get("run_id") or uuid.uuid4().hex
    database = store()
    database.start_run(run_id, state["question"])
    signature = state.get("model_revision") or model_revision()
    return {"run_id": run_id, "model_revision": signature, "reading_recipe": recipe(signature),
            "workflow_revision": WORKFLOW_VERSION, "research_started": time.time()}


def recall(state):
    plan = state.get("news_search_plan", {})
    if "news_api" not in state.get("selected_sources", []) or plan.get("error"):
        return {"memory_documents": []}
    database = store()
    index = MemoryIndex(database)
    docs, info = index.recall(state["question"], state["reading_recipe"], start=plan.get("start", ""),
                              end=plan.get("end", ""),
                              published_after=plan.get("published_after", ""),
                              published_before=plan.get("published_before", ""),
                              section=plan.get("section", ""),
                              source_names=plan.get("source_names", []),
                              limit=settings.memory_recall_k,
                              subject_terms=state.get("subject_terms", []))
    writer()({"kind": "research", "event": "memory_recall", **info})
    return {"memory_documents": docs}


def read_documents(state, *, database=None, read_fn=None, index=None):
    database = database or store()
    emit = writer()
    specs, materials = [], []
    reading_recipe = state["reading_recipe"]
    run_id = state["run_id"]
    for document in state["documents"]:
        if document.metadata.get("source_type") != "news_api":
            materials.append((document, None, None))
            continue
        if document.metadata.get("reading_id") and document.metadata.get("memory_version"):
            # 补搜保留下来的证据可能已经是笔记，必须回到原文版本而不是把笔记当新闻重新阅读。
            screening = {k: v for k, v in document.metadata.items() if k.startswith("relevance_")}
            document = database.document(document.metadata["memory_version"])
            document.metadata.update(screening)
            document.metadata["memory_origin"] = True
        article = database.register(document, settings.reading_chunk_chars)
        for chunk in article["chunks"]:
            chunk["task_key"] = fingerprint(["read", article["id"], article["header"], chunk["hash"], reading_recipe,
                                             len(article["body"]) <= settings.reading_verbatim_max_chars])
        saved = database.reading(article["version"], reading_recipe)
        materials.append((document, article, saved))
        if saved and saved["status"] == "complete":
            event = {"kind": "research", "event": "article_reused", "title": article["metadata"].get("title", ""), "covered": saved["covered"], "total": saved["total"]}
            database.event(run_id, event)
            emit(event)
            continue
        for chunk in article["chunks"]:
            key = chunk["task_key"]
            specs.append(TaskSpec(key, "reader", {"goal": f"阅读《{document.metadata.get('title', '')}》第 {chunk['ordinal'] + 1}/{len(article['chunks'])} 段，提取有出处的事实与观点",
                "header": article["header"], "text": chunk["body"],
                "verbatim": len(article["body"]) <= settings.reading_verbatim_max_chars}))

    def handle(payload):
        if read_fn:
            return validate_reading(read_fn(payload), payload["text"])
        ctx = task_context.get()
        if payload.get("verbatim"):
            if ctx:
                ctx["emit"]("verbatim_read", {"characters": len(payload["text"]), "model_calls": 0})
            return validate_reading({"claims": [{"kind": "source_excerpt", "statement": payload["text"],
                "quote": payload["text"]}], "limitations": ["短原文直接保留，未调用阅读模型、未做独立事实核实；事实分类由后续检查完成。"]}, payload["text"])
        on_split = (lambda sizes: ctx["emit"]("reading_split", {"characters": sizes})) if ctx else None
        return read_segment(payload, on_split=on_split)

    reserve = min(3, settings.specialist_count) if state.get("task_type") != "factual" else 0
    existing = database.tasks(run_id)
    known = {t["key"] for t in existing}
    required = len(existing) + len({s.key for s in specs} - known) + reserve
    # 文章数量在检索端控制；不能因为文章长、分段多就少读原文。
    # 保留 task_budget 字段供旧记录展示，但它现在是需求计数，不再截断阅读队列。
    budget = required
    event = {"kind": "research", "event": "budget_plan", "required": required,
             "allowed": None, "reserved_specialists": reserve,
             "articles": sum(article is not None for _, article, _ in materials),
             "workers": settings.agent_workers}
    database.event(run_id, event)
    emit(event)
    results = Scheduler(database, run_id, limit_tasks=False, emit=emit).run(specs, {"reader": handle}) if specs else {}
    documents, reports = [], []
    for document, article, saved in materials:
        if article is None:
            documents.append(document)
            continue
        if saved is None or saved["status"] != "complete":
            completed = [(chunk, results[chunk["task_key"]]) for chunk in article["chunks"] if results.get(chunk["task_key"])]
            saved = database.save_reading(article, reading_recipe, completed)
        lines = []
        for claim in saved["claims"]:
            lines.append(f"类型：{claim['kind']}；事件背景：{claim.get('event_context', '原文未明确，不得用发布时间代替')}；提取：{claim['statement']}\n原文位置：{claim['start']}–{claim['end']}；原文：{claim['quote']}")
        task_keys = saved.get("task_keys") or [c["task_key"] for c in article["chunks"]
                    if (database.task(c["task_key"]) or {}).get("status") == "complete"]
        report = {"title": document.metadata.get("title", ""), "status": saved["status"], "covered": saved["covered"], "total": saved["total"],
                  "version": article["version"], "reading_id": saved["id"], "task_keys": task_keys}
        reports.append(report)
        # 不完整时显式保留覆盖范围；不把摘要或未读部分标成全文。
        scope = document.metadata.get("content_kind", "news_article")
        body = (f"阅读覆盖：{saved['covered']}/{saved['total']} 段；材料类型：{scope}。以下是来源报道的提取，并非独立核实。\n"
                + "\n".join(lines) + "\n限制：" + "；".join(saved["limitations"]))
        if not lines:
            body += "\n未获得可验证的阅读事实。原始材料片段：\n" + document.page_content[:1000]
        documents.append(Document(page_content=body, metadata={**document.metadata, "memory_version": article["version"],
                         "reading_id": saved["id"], "reading_status": saved["status"],
                         "reading_claims": saved["claims"], "source_text": article["body"],
                         "content_hash": fingerprint([article["version"], body])}))
    sync = (index or MemoryIndex(database)).flush()
    emit({"kind": "research", "event": "index_sync", **sync})
    partial = [report for report in reports if report["status"] != "complete"]
    if partial:
        event = {"kind": "research", "event": "reading_pending", "reports": reports,
                 "covered": sum(r["covered"] for r in reports), "total": sum(r["total"] for r in reports)}
        database.event(run_id, event)
        emit(event)
        raise ResearchWorkPending(f"仍有 {len(partial)} 篇文章未读完（{event['covered']}/{event['total']} 段）；"
                                  "队列和已完成成果已保存，请恢复此研究；失败任务需 --retry-failed。")
    signature = fingerprint(sorted((str(d.metadata.get("source", "")), str(d.metadata.get("memory_version") or fingerprint(d.page_content))) for d in documents))
    unchanged = state.get("evidence_signature") == signature and state.get("retries", 0) > 0
    return {"documents": documents, "task_budget": budget, "research_documents": [material[0] for material in materials],
            "reading_reports": reports, "evidence_signature": signature,
            "no_progress_rounds": state.get("no_progress_rounds", 0) + 1 if unchanged else 0}


def dispatch_specialists(state, *, database=None, analyze_fn=None, index=None):
    if state.get("execution_violations"):
        return {"specialist_findings": [], "specialist_execution": {"expected": 0, "completed": 0}, "next_action": "generate"}
    database = database or store()
    context = state.get("evidence_context", "")
    needs = state.get("evidence_needs", [])[:min(3, settings.specialist_count)]
    if state.get("task_type") == "factual" or not state.get("answer_documents"):
        return {"specialist_findings": [], "next_action": "generate"}
    if not needs:
        needs = ["核对证据之间的因果关系、冲突与条件"]
    specs = []
    for need in needs:
        payload = {"goal": need, "question": state.get("original_question", state["question"]), "context": context,
                   "evidence_versions": [d.metadata.get("memory_version") or fingerprint(d.page_content) for d in state["answer_documents"]]}
        key = fingerprint([SPECIALIST_VERSION, payload, state["model_revision"], datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()])
        dependencies = list(dict.fromkeys(key for report in state.get("reading_reports", [])
                            if report["status"] == "complete" for key in report.get("task_keys", [])))
        specs.append(TaskSpec(key, "specialist", payload, dependencies))
    valid_ids = {f"E{i}" for i in range(1, len(state["answer_documents"]) + 1)}

    def handle(payload):
        response = analyze_fn(payload) if analyze_fn else specialist().invoke(payload)
        result = response.model_dump() if hasattr(response, "model_dump") else response
        passages = []
        requests = result.get("reread_evidence_ids", [])
        if requests:
            ctx = task_context.get()
            emit_tool = (lambda event: ctx["emit"]("tool", event)) if ctx else writer()
            observation = execute_tool(read_news, {"evidence_ids": requests, "goal": payload["goal"]},
                context=ToolContext(
                    emit=emit_tool,
                    caller=f"专题Agent/{str(ctx.get('task_id', 'unknown'))[:12]}" if ctx else "专题Agent",
                    reason=f"回读原文以核对专题任务：{payload['goal']}",
                    read_version=database.document,
                    evidence={f"E{i}": doc for i, doc in enumerate(state["answer_documents"], 1)},
                ))
            passages = observation.artifact.passages
            if ctx:
                ctx["emit"]("reread", {"evidence_ids": requests[:2], "passages": len(passages)})
            details = "\n".join(f"[{p['evidence_id']}] 原文位置 {p['start']}–{p['end']}：{p['quote']}" for p in passages)
            if observation.artifact.warnings:
                details += "\n回读限制：" + "；".join(observation.artifact.warnings)
            if not passages:
                details = "请求的材料没有已保存的新闻全文，只能使用现有摘要；不得伪装成全文。"
            reread_context = format_evidence(state["answer_documents"], payload["goal"], max(1000, settings.generation_context_chars - len(details) - 100))
            next_payload = {**payload, "context": reread_context + "\n原文回读结果（本轮不再回读）：\n" + details}
            response = analyze_fn(next_payload) if analyze_fn else specialist().invoke(next_payload)
            result = response.model_dump() if hasattr(response, "model_dump") else response
        cited = set(result["evidence_ids"]) | set(re.findall(r"\bE\d+\b", result["summary"]))
        if not cited.issubset(valid_ids):
            raise ValueError("专题成果引用了不存在的证据编号")
        result["reread_passages"] = passages
        result["goal"] = payload["goal"]
        result["_vector"] = {"text": payload["goal"] + "\n" + result["summary"],
            "metadata": {"question": payload["question"], "evidence_hash": fingerprint(context), "model_revision": state["model_revision"],
                         "kind": "conditional_analysis", "source_run": state["run_id"]}}
        return result

    results = Scheduler(database, state["run_id"], limit_tasks=False, emit=writer()).run(specs, {"specialist": handle})
    findings = [results[spec.key] for spec in specs if results.get(spec.key)]
    sync = (index or MemoryIndex(database)).flush()
    writer()({"kind": "research", "event": "index_sync", **sync})
    if len(findings) != len(specs):
        raise ResearchWorkPending(f"专题任务仅完成 {len(findings)}/{len(specs)}；已完成成果已保存，"
                                  "请恢复此研究，失败任务需 --retry-failed。")
    from ..graph.nodes import _pending_queries
    gaps = {"news_queries": [], "web_queries": []}
    for finding in findings:
        if finding["needs_more_evidence"]:
            gaps["news_queries"].extend(finding["news_queries"])
            gaps["web_queries"].extend(finding["web_queries"])
    # 未来数值未知不等于缺失历史事实；这只是自动补搜的护栏，不修改用户查询。
    year = datetime.now(ZoneInfo("Asia/Shanghai")).year
    for kind in gaps:
        rejected = [q for q in gaps[kind] if "预测" in q and any(int(y) > year for y in re.findall(r"20\d{2}", q))]
        gaps[kind] = [q for q in gaps[kind] if q not in rejected]
        if rejected:
            writer()({"kind": "research", "event": "search_skipped", "queries": rejected,
                      "reason": "未来预测数值不是必需的已发生事实，保留为不确定性，不因缺少它反复补搜"})
    news, web = _pending_queries(state, gaps)
    can_search = state.get("retries", 0) < settings.research_max_rounds and (news or web)
    # 补搜由轮次、无进展检测和全局时间/模型调用预算控制，不再被阅读分段数阻断。
    can_search = can_search and not state.get("task_contract")  # 不把补搜资料混入用户限定的母集。
    can_search = can_search and state.get("no_progress_rounds", 0) < 2
    passages = list({fingerprint(p): p for f in findings for p in f.get("reread_passages", [])}.values())
    if passages:
        budget = min(max(400, settings.generation_context_chars, len(state["answer_documents"]) * 550),
                     max(400, settings.llm_context_window - min(5000, settings.llm_max_output_tokens) - 2500))
        blocks = [f"[{p['evidence_id']}] 原文回读 {p['start']}–{p['end']}：{p['quote']}" for p in passages]
        # 优先保住每篇已有事实；只按完整回读块追加，不截断句子也不把批量上下文缩回默认值。
        base = format_evidence(state["answer_documents"], state["question"], budget)
        selected = []
        for block in blocks:
            extra = "\n".join([*selected, block])
            if len(extra) > budget // 3:
                continue
            candidate = format_evidence(state["answer_documents"], state["question"], budget - len(extra) - 30)
            if "完整事实块超过上下文预算" not in candidate:
                selected.append(block)
                base = candidate
        context = base + ("\n补充的原文证据：\n" + "\n".join(selected) if selected else "")
        if len(selected) < len(blocks):
            writer()({"kind": "research", "event": "reread_context_limit", "included": len(selected),
                      "total": len(blocks), "reason": "保持全部入选文章的完整事实；其余回读保存在专题成果，未进入最终上下文"})
    return {"specialist_findings": findings, "specialist_execution": {"expected": len(specs), "completed": len(findings)},
            "evidence_context": context, "pending_news_queries": news, "pending_web_queries": web,
            "next_action": "supplement" if can_search else "generate"}
