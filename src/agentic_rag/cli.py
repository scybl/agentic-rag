"""命令行界面：逐步展示可验证的智能体执行轨迹。"""

import argparse
import logging
import os
import sys
import time
import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ollama import Client, ResponseError

# 关闭模型库进度条，让终端聚焦于智能体步骤。必须在加载嵌入模型前设置。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .config import settings
from .graph.build import build_graph
from .ingestion import start_knowledge_watcher
from .ollama_connection import ollama_client_kwargs


SOURCE_LABELS = {
    "vectorstore": "本地知识库",
    "news_api": "新闻 API",
    "web_search": "公开网络",
}
SOURCE_STEPS = {"vectorstore": "2", "news_api": "3", "web_search": "4"}


def _clip(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _display(value: Any) -> str:
    """完整展示决策和查询字段，只折叠换行，不截断关键信息。"""
    return " ".join(str(value or "").split())


def _time_range(start: str = "", end: str = "") -> str:
    if not start and not end:
        return "不限时间（不发送 start/end）"
    return f"{start or '不限起点'} 至 {end or '不限终点'}"


def warm_up_model(*, client: Client | None = None, sleep_fn=time.sleep) -> bool:
    """在接收问题前加载模型；本机 Ollama 直接连接，不继承系统代理。"""
    ollama_client = client or Client(
        host=settings.ollama_base_url,
        **ollama_client_kwargs(settings.ollama_base_url),
    )
    attempts = max(1, settings.ollama_warmup_attempts)
    started = time.perf_counter()
    print(f"正在加载本地模型 {settings.llm_model}，首次可能需要几十秒……", flush=True)
    for attempt in range(1, attempts + 1):
        try:
            ollama_client.generate(
                model=settings.llm_model,
                prompt="",
                stream=False,
                think=False,
                keep_alive=settings.ollama_keep_alive,
                options={"num_ctx": settings.llm_context_window, "num_predict": 1},
            )
            elapsed = time.perf_counter() - started
            print(f"模型已就绪（{elapsed:.1f} 秒）", flush=True)
            return True
        except ResponseError as exc:
            if getattr(exc, "status_code", None) != 503 or attempt >= attempts:
                print(f"模型预热失败：{_run_error_message(exc)}", flush=True)
                return False
        except Exception as exc:
            if attempt >= attempts:
                print(f"模型预热失败：{_run_error_message(exc)}", flush=True)
                return False
        delay = max(0.0, settings.ollama_warmup_retry_seconds)
        print(
            f"模型暂未就绪，{delay:g} 秒后重试（{attempt + 1}/{attempts}）……",
            flush=True,
        )
        sleep_fn(delay)
    return False


class TracePrinter:
    """把图节点更新转换成适合学习的执行摘要，不展示隐藏思维链。"""

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.round = 0

    @staticmethod
    def _heading(flow_step: str, title: str) -> None:
        print(f"\n[流程 {flow_step} · {title}]", flush=True)

    def show_event(self, event: dict[str, Any]) -> None:
        """实时显示已提交的 API 参数和观测结果；事件不含认证请求头。"""
        kind = event.get("kind")
        if kind == "research":
            name = event.get("event")
            labels = {"queued": "排队", "started": "已领取", "waiting_llm": "等待模型槽位",
                      "running_llm": "模型请求执行中", "waiting_io": "等待网络槽位", "running_io": "网络请求执行中",
                      "completed": "完成", "reused": "复用已完成成果", "attempt_failed": "本次尝试失败",
                      "failed": "失败", "deferred": "暂缓", "blocked": "等待依赖", "budget": "达到任务预算", "timeout": "达到时间预算"}
            if name == "memory_recall":
                print(f"  [长期记忆] 命中 {event['count']} 篇；语义检索：{'已使用' if event['semantic'] else '未使用'}；仍检查新闻源更新")
            elif name == "article_reused":
                print(f"  [阅读复用] {event['title']}：{event['covered']}/{event['total']} 段，无需重新调用模型")
            elif name == "index_sync":
                print(f"  [向量投影] 已同步 {event['synced']}；失败 {event['failed']}；待同步 {event['remaining']}（失败不丢阅读成果）")
            elif name == "search_skipped":
                print(f"  [补搜约束] 不执行：{event['queries']}；原因：{event['reason']}")
            elif name == "reread":
                print(f"  [任务 {event.get('task_id', '')} · 原文回读] 请求 {event['evidence_ids']}，返回 {event['passages']} 个可追溯片段")
            else:
                print(f"  [任务 {event.get('task_id', '')} · {event.get('role', '')}] {labels.get(name, name)}")
                if name == "queued":
                    print(f"    指令：{event.get('goal', '')}；依赖：{event.get('dependencies', [])}")
                if "active" in event:
                    print(f"    进程内在途请求：{event['active']}/{event['limit']}（服务端仍可能排队）")
            if event.get("error"):
                print(f"    提醒：{_display(event['error'])}")
        elif kind == "news_request":
            self._heading("3", f"新闻 API · 第 {event['page']} 页请求")
            if event.get("query_total", 1) > 1:
                print(f"  查询组：{event['query_number']}/{event['query_total']}")
            print(f"  关键词 q：{event.get('query')!r}（空字符串表示不按关键词筛选）")
            print(f"  实际时间：{_time_range(event.get('start', ''), event.get('end', ''))}")
            print(f"  栏目：{event.get('section') or '不限'}")
            order = "最早在前" if event.get("order") == "asc" else "最新在前"
            print(f"  本页上限：{event['limit']} 条；排序：{order}；续页：{'是' if event.get('continuation') else '否'}")
            if "max_items" in event:
                print(f"  本组候选上限：{event['max_items'] if event['max_items'] is not None else '不限'}；本轮总上限：{event.get('candidate_limit', event['max_items'])}")
        elif kind == "news_page":
            print(f"  第 {event['page']} 页返回：{event['count']} 条；还有下一页：{'是' if event['has_more'] else '否'}")
        elif kind == "news_error":
            print(f"  新闻 API 请求失败：{_display(event['message'])}")
            if event.get("partial_count"):
                print(f"  保留已取得的 {event['partial_count']} 条候选，继续处理")
        elif kind == "news_cache_fallback":
            if event.get("date_restricted"):
                print("  本机缓存未按发布时间检索，无法保证符合本次日期条件，未采用缓存证据")
            else:
                print(f"  本机新闻缓存回退：{event['count']} 条（不是本次 API 返回）")
        elif kind == "news_ranked":
            print(f"  新闻候选：{event['candidate_count']} 条 → 入选：{event['selected_count']} 条")
            if "raw_count" in event:
                print(f"  多组原始返回：{event['raw_count']} 条；按文章 ID 去重后：{event['candidate_count']} 条")
            print(f"  向量缓存/排序：{'已使用' if event['vector_cache_used'] else '未使用，保持 API 顺序'}")
            if event.get("cache_stats") is not None:
                print(f"  向量缓存更新：{event['cache_stats']}")
        elif kind == "news_article_error":
            print(f"  文章 {event['article_id']} 正文读取失败，使用已有摘要：{_display(event['message'])}")
        sys.stdout.flush()

    def show(self, node: str, update: dict[str, Any], before: dict[str, Any]) -> None:
        if node == "initialize_research":
            self._heading("0", "建立可恢复研究")
            print(f"  研究编号：{update['run_id']}；工作任务上限：{settings.agent_workers}；模型请求上限：{settings.llm_concurrency}")
            return
        if node == "read_documents":
            self._heading("5b", "独立阅读与成果持久化")
            for report in update.get("reading_reports", []):
                print(f"  {report['title']}：{report['covered']}/{report['total']} 段；{report['status']}")
                print(f"    阅读成果编号：{report['reading_id']}")
            return
        if node == "dispatch_specialists":
            self._heading("7b", "汇总专题 Agent")
            for result in update.get("specialist_findings", []):
                print(f"  指令：{result['goal']}\n  结果：{result['summary']}")
                if result.get("limitations"):
                    print(f"  限制：{'；'.join(result['limitations'])}")
            print(f"  下一步：{'追加补搜任务' if update.get('next_action') == 'supplement' else '生成综合答案'}")
            if update.get("next_action") == "supplement":
                self._show_pending_queries(update)
            return
        if node == "route":
            self.round += 1
            self._heading("1", f"规划数据源（第 {self.round} 轮）")
            selected = update.get("selected_sources", [])
            labels = [SOURCE_LABELS.get(source, source) for source in selected]
            print(f"  选择：{' + '.join(labels) if labels else '未选择'}")
            if update.get("task_type"):
                task_label = {"forecast": "预测", "analysis": "分析", "factual": "事实问答"}.get(update["task_type"], update["task_type"])
                print(f"  任务类型：{task_label}；需要核对：{'、'.join(update.get('evidence_needs', []))}")
            summary = _display(update.get("plan_summary"))
            if summary:
                print(f"  原因：{summary}")
            for source, query in update.get("source_queries", {}).items():
                print(f"  查询/{SOURCE_LABELS.get(source, source)}：{_display(query) or '（空：不按关键词筛选）'}")
            plan = update.get("news_search_plan", {})
            if plan:
                if plan.get("raw_queries") and plan["raw_queries"] != plan.get("queries"):
                    print(f"  模型原始新闻词：{plan['raw_queries']}")
                    print(f"  查询整理：拆分分号列表、去重，最多保留 4 组；实际执行：{plan['queries']}")
                suggestion = plan.get("suggested_time", {})
                mode = {"unrestricted": "不限时间", "suggested": "模型建议", "explicit": "用户指定"}.get(suggestion.get("mode"), "未提供")
                print(f"  模型时间选择：{mode}")
                print(f"  模型给出的日期：{_time_range(suggestion.get('start', ''), suggestion.get('end', ''))}")
                print(f"  时间说明：{_display(suggestion.get('reason')) or '未提供'}")
                actual_time = "不执行查询（日期无效）" if plan.get("error") else _time_range(plan.get('start', ''), plan.get('end', ''))
                print(f"  实际采用日期：{actual_time}")
                print(f"  时间校验：{_display(plan.get('time_note'))}")
            sys.stdout.flush()
            return

        if node in {"collect_sources", "supplement_sources"}:
            self._heading("2–5" if node == "collect_sources" else "7", "收集并合并证据" if node == "collect_sources" else "补充缺失证据")
            grouped = update.get("documents_by_source", {})
            for source in dict.fromkeys([*before.get("selected_sources", []), *grouped]):
                count = len(grouped.get(source, []))
                step = SOURCE_STEPS.get(source, "?")
                print(f"  [{step}] {SOURCE_LABELS.get(source, source)}：{count} 条")
            print(f"  [5] 合并去重后：{len(update.get('documents', []))} 条")
            errors = update.get("source_errors", {})
            for source, error in errors.items():
                message = f"（{_display(error)}）"
                print(f"  提醒：{SOURCE_LABELS.get(source, source)}发生查询失败{message}，保留已取得的结果并继续")
            if node == "supplement_sources":
                print(f"  已补搜：{update.get('retries', 0)}/{settings.research_max_rounds if before.get('reading_recipe') else settings.max_retries} 轮；旧证据已保留")
            return

        if node == "grade_documents":
            self._heading("6", "筛选相关证据")
            total = len(before.get("documents", []))
            kept = len(update.get("documents", []))
            print(f"  保留：{kept}/{total} 条")
            for source, docs in before.get("documents_by_source", {}).items():
                remaining = len(update.get("documents_by_source", {}).get(source, []))
                print(f"  {SOURCE_LABELS.get(source, source)}：{len(docs)} → {remaining} 条")
            print("  决策：进入整批证据充分性检查")
            return

        if node == "assess_evidence":
            self._heading("6b", "检查证据是否足以分析")
            review = update.get("evidence_assessment", {})
            print(f"  可进行条件分析：{'是' if review.get('ready') else '证据仍有缺口'}")
            print(f"  判断：{_display(review.get('summary'))}")
            for factor in review.get("covered_factors", []):
                print(f"  已覆盖：{_display(factor)}")
            for factor in review.get("missing_factors", []):
                print(f"  缺口：{_display(factor)}")
            print(f"  下一步：{'针对缺口补搜' if update.get('next_action') == 'supplement' else '依据现有证据生成，并说明限制'}")
            self._show_pending_queries(update, suggested=update.get("next_action") != "supplement")
            return

        if node in {"generate", "revise_answer"}:
            self._heading("8", "生成答案" if node == "generate" else "按反思反馈修正分析")
            hit = bool(update.get("analysis_cache_hit"))
            print(f"  分析缓存：{'命中，复用已核验答案' if hit else '未命中，依据当前证据生成'}")
            used = update.get("answer_documents", before.get("documents", []))
            print(f"  使用证据：{len(used)} 条（已按模型上下文预算截取）")
            print(f"  实际证据片段：{len(update.get('evidence_context', ''))} 字符")
            for index, document in enumerate(used, 1):
                metadata = document.metadata
                title = metadata.get("title") or metadata.get("source", "unknown")
                published = metadata.get("published_at") or "未标注日期"
                print(f"    - [E{index}] {_display(title)} | {published}")
            return

        if node == "evaluate_generation":
            self._heading("9", "核验答案依据")
            grounded = bool(update.get("generation_grounded"))
            complete = bool(update.get("generation_complete"))
            print(f"  事实与推断依据：{'通过' if grounded else '未通过'}")
            print(f"  是否完成用户任务：{'通过' if complete else '未通过'}")
            print(f"  决策：{_display(update.get('generation_check'))}")
            action = update.get("next_action", "finish")
            action_label = {"finish": "结束", "revise": "依据反馈重写", "supplement": "补充证据"}.get(action, action)
            print(f"  下一步：{action_label}")
            if action != "finish":
                print(f"  修正要求：{_display(update.get('revision_feedback'))}")
                if action == "supplement":
                    self._show_pending_queries(update)

    @staticmethod
    def _show_pending_queries(update: dict[str, Any], *, suggested: bool = False) -> None:
        label = "建议补搜（本轮不执行）" if suggested else "待执行补搜"
        for query in update.get("pending_news_queries", []):
            print(f"  {label}/新闻：{query}")
        for query in update.get("pending_web_queries", []):
            print(f"  {label}/网络：{query}")


def run_with_trace(graph, question: str, *, verbose: bool = False, run_id=None, resume=False) -> dict[str, Any]:
    """流式运行图并合并节点更新，返回与 graph.invoke 相同用途的最终状态。"""
    state: dict[str, Any] = {"question": question}
    trace = TracePrinter(verbose=verbose)
    input_state = {"question": question}
    options = {}
    if run_id:
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        options["config"] = config
        input_state["run_id"] = run_id
        if resume:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise ValueError("没有找到可恢复的检查点")
            state = dict(snapshot.values)
            if not snapshot.next:
                print("  该研究已完成，展示保存的最终结果。")
                return state
            if state.get("model_revision"):
                from .research.service import model_revision, recipe, WORKFLOW_VERSION
                if (state["model_revision"] != model_revision()
                        or state.get("reading_recipe") != recipe(state["model_revision"])
                        or state.get("workflow_revision") != WORKFLOW_VERSION):
                    raise ValueError("模型版本或上下文配置已变化，请发起新研究，避免混用旧检查点")
            input_state = None
    for mode, event in graph.stream(input_state, stream_mode=["updates", "custom"], **options):
        if mode == "custom":
            trace.show_event(event)
            continue
        for node, update in event.items():
            if not isinstance(update, dict):
                continue
            before = dict(state)
            trace.show(node, update, before)
            state.update(update)
    return state


def _run_error_message(exc: BaseException) -> str:
    """把常见的 Ollama 异常转换成可操作的终端提示。"""
    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None)
        if status_code == 503:
            return (
                "模型服务请求返回 HTTP 503，重试后仍失败。"
                "请检查 Ollama 服务日志；仅凭状态码无法确定是否是加载或连接问题。"
            )
        name = type(current).__name__
        if name in {"ConnectError", "ConnectionError"}:
            return "无法连接 Ollama。请确认 Ollama 正在运行，并检查 OLLAMA_BASE_URL。"
        if name == "EmptyModelOutputError":
            return "Ollama 多次返回空文本。本次结果不会缓存，请直接重试或换用较小模型。"
        current = current.__cause__ or current.__context__
    return f"智能体执行失败：{type(exc).__name__}: {_clip(exc, 160)}"


def ask(graph, question: str, *, verbose: bool = False, run_id=None, resume=False, retry_failed=False) -> bool:
    print(f"\n问题：{question}")
    database = None
    acquired = False
    try:
        with ExitStack() as stack:
            database = None
            if run_id:
                from .research.service import store
                from .research.session import run_lease
                database = store()
                if resume:
                    snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
                    if not snapshot.values:
                        raise ValueError("没有找到可恢复的检查点")
                    question = snapshot.values.get("question", "")
                    if snapshot.values.get("model_revision") and (snapshot.next or retry_failed):
                        from .research.service import model_revision, recipe, WORKFLOW_VERSION
                        if (snapshot.values["model_revision"] != model_revision()
                                or snapshot.values.get("reading_recipe") != recipe(snapshot.values["model_revision"])
                                or snapshot.values.get("workflow_revision") != WORKFLOW_VERSION):
                            raise ValueError("模型版本或配置已变化，请发起新研究")
                stack.enter_context(run_lease(database, run_id, question))
                acquired = True
                if retry_failed:
                    count = database.retry_failed(run_id)
                    if count:
                        graph.update_state({"configurable": {"thread_id": run_id}},
                            {"documents": snapshot.values.get("research_documents", snapshot.values.get("documents", []))}, as_node="collect_sources")
                    print(f"已重置 {count} 个失败任务的重试预算")
                database.run_status(run_id, "running")
                print(f"  研究编号：{run_id}；中断后用 agentic-rag --resume {run_id} 恢复", flush=True)
            result = run_with_trace(graph, question, verbose=verbose, run_id=run_id, resume=resume)
            if database:
                database.run_status(run_id, "completed" if result.get("generation_complete") and result.get("generation_grounded") else "needs_attention")
    except KeyboardInterrupt:
        if run_id and database and acquired:
            database.run_status(run_id, "interrupted")
            print(f"\n研究已中断，已完成成果保留。恢复：agentic-rag --resume {run_id}")
        return False
    except Exception as exc:
        if run_id and database and acquired:
            database.run_status(run_id, "failed")
        print("\n[本次执行未完成]")
        print(f"  {_run_error_message(exc)}")
        print("  交互模式仍可继续，请直接重新输入问题。")
        if run_id:
            print(f"  恢复：agentic-rag --resume {run_id}")
        if verbose:
            print(f"  异常类型：{type(exc).__module__}.{type(exc).__name__}")
        return False
    print(f"\n{'=' * 18} 最终回答 {'=' * 18}\n")
    print(result["generation"])
    sources = {
        doc.metadata.get("source", "unknown")
        for doc in result.get("answer_documents", result.get("documents", []))
    }
    if sources:
        used_sources = sorted({
            doc.metadata.get("source_type", "unknown")
            for doc in result.get("answer_documents", result.get("documents", []))
        })
        print(f"\n[实际使用数据源：{'+'.join(used_sources)}]")
        for index, document in enumerate(result.get("answer_documents", result.get("documents", [])), 1):
            print(f"  [E{index}] {document.metadata.get('source', 'unknown')}")
    return True


def main() -> None:
    # VS Code 终端使用 UTF-8；显式设置可避免 Windows 重定向输出时出现乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Multi-source agentic RAG")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--resume", metavar="RUN_ID", help="恢复已保存研究；不与新问题同时使用")
    parser.add_argument("--runs", action="store_true", help="列出最近研究，不调用模型")
    parser.add_argument("--status", metavar="RUN_ID", help="查看任务与最近执行事件")
    parser.add_argument("--retry-failed", action="store_true", help="与 --resume 配合，重新尝试失败子任务")
    parser.add_argument("--inspect-reading", metavar="READING_ID", help="查看阅读成果、原文位置与版本")
    parser.add_argument("--repair-memory", action="store_true", help="仅重试向量投影，不重新阅读")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show additional exception details; queries and time choices are always shown",
    )
    args = parser.parse_args()
    if args.resume and args.question:
        parser.error("--resume 不能同时提交新问题")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed 需要 --resume")
    from .research.service import store
    database = store()
    if args.runs or args.status or args.inspect_reading or args.repair_memory:
        if args.runs:
            print(json.dumps(database.runs(), ensure_ascii=False, indent=2))
        if args.status:
            print(json.dumps({"tasks": database.tasks(args.status), "events": database.events(args.status)}, ensure_ascii=False, indent=2))
        if args.inspect_reading:
            print(json.dumps(database.inspect_reading(args.inspect_reading), ensure_ascii=False, indent=2))
        if args.repair_memory:
            from .research.memory import MemoryIndex
            index = MemoryIndex(database)
            while True:
                result = index.flush()
                print(json.dumps(result, ensure_ascii=False))
                if result["failed"] or not result["remaining"]:
                    break
        return

    # 终端使用结构化步骤追踪；第三方库只保留错误，避免 HTTP 和回退警告淹没主流程。
    logging.basicConfig(level=logging.ERROR, format="  %(message)s")

    watcher = None
    try:
        if settings.knowledge_watch_enabled:
            watcher = start_knowledge_watcher()

        if settings.ollama_warmup_enabled:
            if not warm_up_model():
                print("Ollama 模型未就绪，程序停止；请检查 Ollama 后重新启动。")
                raise SystemExit(1)

        from langgraph.checkpoint.sqlite import SqliteSaver
        Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            saver = resources.enter_context(SqliteSaver.from_conn_string(settings.checkpoint_db))
            graph = build_graph(checkpointer=saver)

            if args.resume:
                succeeded = ask(graph, "", verbose=args.verbose, run_id=args.resume, resume=True, retry_failed=args.retry_failed)
                if not succeeded:
                    raise SystemExit(1)
                return

            if args.question:
                succeeded = ask(graph, " ".join(args.question), verbose=args.verbose, run_id=uuid.uuid4().hex)
                if not succeeded:
                    raise SystemExit(1)
                return

            print("Agentic RAG — 交互模式（Ctrl+C 或输入 exit 退出）")
            while True:
                try:
                    question = input("\n> ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if not question or question.lower() in {"exit", "quit"}:
                    break
                ask(graph, question, verbose=args.verbose, run_id=uuid.uuid4().hex)
    finally:
        if watcher is not None:
            watcher.stop()
            watcher.join()


if __name__ == "__main__":
    main()
