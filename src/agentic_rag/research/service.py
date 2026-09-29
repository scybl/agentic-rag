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
from .chains import reader, specialist, validate_reading, numbered_sentences, resolve_selection
from .memory import MemoryIndex
from .scheduler import Scheduler, TaskSpec
from .store import ResearchStore, fingerprint
from .runtime import task_context
from ..evidence import format_evidence
from ..tools import read_news
from ..tools.contracts import ToolContext
from ..tools.execution import execute_tool


READING_VERSION = "reader-v2-sentence-selection-schema2"
SPECIALIST_VERSION = "specialist-v4-tool-reread"
WORKFLOW_VERSION = "research-workflow-v2-tools"


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
            return fingerprint([name, model.digest, settings.temperature, settings.llm_context_window, settings.llm_max_output_tokens])
    raise RuntimeError("无法确认本地模型版本，停止建立可能失效的阅读缓存")


def recipe(signature):
    return fingerprint([READING_VERSION, signature, settings.reading_chunk_chars])


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
                              end=plan.get("end", ""), limit=settings.memory_recall_k)
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
            document = database.document(document.metadata["memory_version"])
            document.metadata["memory_origin"] = True
        article = database.register(document, settings.reading_chunk_chars)
        for chunk in article["chunks"]:
            chunk["task_key"] = fingerprint(["read", article["id"], article["header"], chunk["hash"], reading_recipe])
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
                "header": article["header"], "text": chunk["body"]}))

    def handle(payload):
        if read_fn:
            return validate_reading(read_fn(payload), payload["text"])
        sentences, text = numbered_sentences(payload["text"])
        result = reader().invoke({**payload, "text": text})
        return resolve_selection(result, payload["text"], sentences)

    results = Scheduler(database, run_id, emit=emit).run(specs, {"reader": handle}) if specs else {}
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
            lines.append(f"类型：{claim['kind']}；提取：{claim['statement']}\n原文位置：{claim['start']}–{claim['end']}；原文：{claim['quote']}")
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
                         "reading_id": saved["id"], "reading_status": saved["status"], "content_hash": fingerprint([article["version"], body])}))
    sync = (index or MemoryIndex(database)).flush()
    emit({"kind": "research", "event": "index_sync", **sync})
    signature = fingerprint(sorted((str(d.metadata.get("source", "")), str(d.metadata.get("memory_version") or fingerprint(d.page_content))) for d in documents))
    unchanged = state.get("evidence_signature") == signature and state.get("retries", 0) > 0
    return {"documents": documents, "research_documents": [material[0] for material in materials],
            "reading_reports": reports, "evidence_signature": signature,
            "no_progress_rounds": state.get("no_progress_rounds", 0) + 1 if unchanged else 0}


def dispatch_specialists(state, *, database=None, analyze_fn=None, index=None):
    database = database or store()
    context = state.get("evidence_context", "")
    needs = state.get("evidence_needs", [])[:settings.specialist_count]
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
                context=ToolContext(emit=emit_tool, read_version=database.document,
                    evidence={f"E{i}": doc for i, doc in enumerate(state["answer_documents"], 1)}))
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

    results = Scheduler(database, state["run_id"], emit=writer()).run(specs, {"specialist": handle})
    findings = [results[spec.key] for spec in specs if results.get(spec.key)]
    sync = (index or MemoryIndex(database)).flush()
    writer()({"kind": "research", "event": "index_sync", **sync})
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
    # 任务预算耗尽不继续派一轮注定无法阅读的新材料。
    can_search = can_search and len(database.tasks(state["run_id"])) < settings.research_max_tasks
    can_search = can_search and state.get("no_progress_rounds", 0) < 2
    passages = list({fingerprint(p): p for f in findings for p in f.get("reread_passages", [])}.values())
    if passages:
        extra = "\n".join(f"[{p['evidence_id']}] 原文回读 {p['start']}–{p['end']}：{p['quote']}" for p in passages)[:settings.generation_context_chars // 3]
        context = format_evidence(state["answer_documents"], state["question"], settings.generation_context_chars - len(extra) - 30)
        context += "\n补充的原文证据：\n" + extra
    return {"specialist_findings": findings, "evidence_context": context, "pending_news_queries": news, "pending_web_queries": web,
            "next_action": "supplement" if can_search else "generate"}
