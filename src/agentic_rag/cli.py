"""命令行界面：逐步展示可验证的智能体执行轨迹。"""

import argparse
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
from ollama import Client

# 关闭模型库进度条，让终端聚焦于智能体步骤。必须在加载嵌入模型前设置。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .config import settings
from .graph.build import build_graph
from .ingestion import start_knowledge_watcher
from .ollama_connection import ollama_client_kwargs, warmup_failure
from .token_usage import UsageLedger, active_usage, usage_session, print_usage_event, print_compact_usage_event, print_usage_summary, format_duration
from .console import CompactTrace
from .research.inspection import has_pending_work, recoverable_reading, inspect_research, print_inspection


SOURCE_LABELS = {
    "vectorstore": "本地知识库",
    "news_api": "新闻 API",
    "web_search": "公开网络",
}
SOURCE_STEPS = {"vectorstore": "2", "news_api": "3", "web_search": "4"}
DEFAULT_IDLE_TIMEOUT_SECONDS = 5 * 60


class ConsoleInputTimeout(TimeoutError):
    """交互终端在限定时间内没有收到一整行输入。"""


def input_with_timeout(prompt: str, timeout_seconds: float, *, input_fn=input) -> str:
    """保留普通 input 的行编辑体验，同时允许主线程等待超时。"""
    if timeout_seconds <= 0:
        return input_fn(prompt)

    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def read_line() -> None:
        try:
            result.put((True, input_fn(prompt)))
        except BaseException as exc:  # KeyboardInterrupt/EOFError 也应按原语义交还主线程
            result.put((False, exc))

    threading.Thread(target=read_line, name="agentic-rag-console-input", daemon=True).start()
    try:
        succeeded, value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise ConsoleInputTimeout from exc
    if succeeded:
        return str(value)
    raise value


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


def warm_up_model(*, client: Client | None = None, sleep_fn=time.sleep, session_ledger=None, verbose=False) -> bool:
    """在接收问题前加载模型；本机 Ollama 直接连接，不继承系统代理。"""
    ledger = UsageLedger(observer=session_ledger.observe if session_ledger else None)
    try:
        with usage_session(ledger, scope="warmup"):
            return _warm_up_model(client=client, sleep_fn=sleep_fn, ledger=ledger)
    finally:
        print_usage_summary(ledger, title="模型预热 Token（不计入单次问题）", compact=not verbose)


def _warm_up_model(*, client, sleep_fn, ledger):
    if client is None:
        # SDK 默认 timeout=None 会无限等待；连接与加载采用不同的超时上限。
        timeout = max(1, settings.llm_request_timeout)
        try:
            owned_client = Client(host=settings.ollama_base_url,
                                  timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                                  **ollama_client_kwargs(settings.ollama_base_url))
        except Exception as exc:
            _, reason = warmup_failure(exc, settings.ollama_base_url)
            print(f"模型预热初始化失败：{reason}", flush=True)
            return False
        with owned_client as opened_client:
            return _warm_up_model(client=opened_client, sleep_fn=sleep_fn, ledger=ledger)
    ollama_client = client
    attempts = max(1, settings.ollama_warmup_attempts)
    started = time.perf_counter()
    print(f"正在连接 Ollama 并请求预热模型 {settings.llm_model}，首次加载可能需要几十秒……", flush=True)
    for attempt in range(1, attempts + 1):
        attempt_started = time.perf_counter()
        call_id = ledger.begin_call(step={"step": "warmup", "step_id": "warmup"},
                                    operation=f"加载模型（第 {attempt} 次）", thinking_requested=False)
        try:
            response = ollama_client.generate(
                model=settings.llm_model,
                prompt="",
                stream=False,
                think=False,
                keep_alive=settings.ollama_keep_alive,
                options={"num_ctx": settings.llm_context_window, "num_predict": 1},
            )
            ledger.finish_call(call_id, metadata=response.model_dump() if hasattr(response, "model_dump") else response,
                               elapsed=round(time.perf_counter() - attempt_started, 3))
            elapsed = time.perf_counter() - started
            print(f"模型已就绪（{elapsed:.1f} 秒）", flush=True)
            return True
        except Exception as exc:
            ledger.finish_call(call_id, failed=True, elapsed=round(time.perf_counter() - attempt_started, 3))
            retryable, reason = warmup_failure(exc, settings.ollama_base_url)
            if not retryable or attempt >= attempts:
                suffix = " 已达到预热尝试上限。" if retryable else ""
                print(f"模型预热失败：{reason}{suffix}", flush=True)
                return False
        delay = max(0.0, settings.ollama_warmup_retry_seconds)
        print(
            f"预热暂时失败：{reason}\n{delay:g} 秒后重试（{attempt + 1}/{attempts}）……",
            flush=True,
        )
        sleep_fn(delay)
    return False


def unload_model(*, client: Client | None = None, verbose: bool = False) -> bool:
    """通知 Ollama 立即卸载当前模型；退出清理失败不能遮盖原任务结果。"""
    try:
        if client is None:
            with Client(host=settings.ollama_base_url, timeout=httpx.Timeout(5.0),
                        **ollama_client_kwargs(settings.ollama_base_url)) as owned_client:
                return unload_model(client=owned_client, verbose=verbose)
        client.generate(
            model=settings.llm_model,
            prompt="",
            stream=False,
            keep_alive=0,
        )
        print(f"已释放本地模型 {settings.llm_model}。", flush=True)
        return True
    except Exception as exc:
        if verbose:
            print(f"本地模型卸载通知失败：{type(exc).__name__}；可在 Ollama 中检查模型状态。", flush=True)
        return False


class TracePrinter:
    """把图节点更新转换成适合学习的执行摘要，不展示隐藏思维链。"""

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose
        self.round = 0
        self.compact = None if verbose else CompactTrace()
        self.screen_trace = CompactTrace()

    @staticmethod
    def _heading(flow_step: str, title: str) -> None:
        print(f"\n[流程 {flow_step} · {title}]", flush=True)

    def show_event(self, event: dict[str, Any]) -> None:
        """实时显示已提交的 API 参数和观测结果；事件不含认证请求头。"""
        if self.compact is not None:
            return self.compact.show_event(event)
        kind = event.get("kind")
        if kind in {"news_screen_progress", "news_screen_batch", "news_core_replaced", "news_core_ready"}:
            return self.screen_trace.show_event(event)
        if kind == "document_grade":
            label = {"keep": "保留", "exclude": "排除", "uncertain": "待确认，暂存"}.get(event.get("decision"), "未知")
            print(f"  [精读前筛选 · {label}] {_display(event.get('title'))}")
            print(f"    理由：{_display(event.get('reason'))}")
        elif kind == "tool":
            call_id = event.get("tool_call_id", "")[:12]
            print(f"  [工具 {event.get('tool', '')} · {call_id}]")
            if event.get("phase") == "started":
                print(f"    调用者：{_display(event.get('caller'))}")
                print(f"    使用原因：{_display(event.get('reason'))}")
                print(f"    调用参数：{json.dumps(event.get('arguments', {}), ensure_ascii=False)}")
            elif event.get("phase") == "finished":
                labels = {"ok": "成功", "empty": "正常零结果", "degraded": "降级返回",
                          "error": "失败", "rejected": "规则未通过",
                          "adjusted": "动态调整", "redirected": "改用其他策略",
                          "stopped": "停止重试"}
                print(f"    结果：{labels.get(event.get('status'), event.get('status'))}；证据 {event.get('count', 0)} 条；耗时 {event.get('elapsed', 0):.3f} 秒")
                if event.get("action"):
                    changes = event.get("adjustments", {})
                    print(f"    调整动作：{event['action']}；是否重试：{'是' if event.get('retry') else '否'}")
                    print(f"    下一参数：输出 {changes.get('num_predict')}；上下文 {changes.get('num_ctx')}；"
                          f"超时 {changes.get('timeout_seconds')} 秒；思考 {'开启' if changes.get('reasoning') else '关闭'}")
                    print(f"    决策原因：{_display(event.get('plan_reason'))}")
                for check in event.get("checks", []):
                    print(f"    规则/{check.get('name')}：{'通过' if check.get('passed') else '未通过'}；{_display(check.get('detail'))}")
                for warning in event.get("warnings", []):
                    print(f"    提醒：{_display(warning)}")
            else:
                print(f"    失败：{event.get('error_type', '')}；{_display(event.get('error', ''))}")
        elif kind == "research":
            name = event.get("event")
            labels = {"queued": "排队", "started": "已领取", "waiting_llm": "等待模型槽位",
                      "running_llm": "模型请求执行中", "waiting_io": "等待网络槽位", "running_io": "网络请求执行中",
                      "completed": "完成", "reused": "复用已完成成果", "attempt_failed": "本次尝试失败",
                      "failed": "失败", "deferred": "暂缓", "blocked": "等待依赖", "budget": "达到任务预算", "timeout": "达到时间预算"}
            if name == "memory_recall":
                print(f"  [长期记忆] 候选 {event['count']} 篇（仍须筛选）；语义检索：{'已使用' if event['semantic'] else '未使用'}；实体过滤 {event.get('subject_filtered', 0)} 篇")
            elif name == "article_reused":
                print(f"  [阅读复用] {event['title']}：{event['covered']}/{event['total']} 段，无需重新调用模型")
            elif name == "reading_split":
                print(f"  [阅读缩段] 输出截断，保留思考并拆成两段：{event['characters']} 字符；最多拆一次，不原样重试")
            elif name == "budget_plan":
                print(f"  [任务计划] 需要 {event['required']} / 数量上限 {event['allowed'] or '不限'}；专题预留 {event['reserved_specialists']}；并发 {event.get('workers', '?')}")
            elif name == "reread_context_limit":
                print(f"  [回读上下文] 已纳入 {event['included']}/{event['total']} 块；{event['reason']}")
            elif name == "specialist_advice_filtered":
                print(f"  [专题建议校验] {event['goal']}；未采纳相关句子：{event['reason']}")
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
                if name == "completed" and "elapsed" in event:
                    print(f"    任务耗时：{format_duration(event['elapsed'])}（含执行中的等待/重试，不含线程池排队）")
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
                print(f"  本组候选上限：{event['max_items'] if event['max_items'] is not None else '不限，持续取页直到结束'}")
        elif kind == "news_retry":
            print(f"  新闻临时错误 HTTP {event['status_code']}；{event['delay']} 秒后第 {event['attempt']}/{event['max_attempts']} 次尝试；q={event.get('query')!r}")
            if event.get("request_id"):
                print(f"  服务请求编号：{event['request_id']}；分类：{event.get('error_code') or '未标注'}")
        elif kind == "news_page":
            print(f"  第 {event['page']} 页返回：{event['count']} 条；还有下一页：{'是' if event['has_more'] else '否'}")
            if event.get("request_id"):
                print(f"  服务请求编号：{event['request_id']}")
        elif kind == "news_error":
            print(f"  新闻 API 请求失败：{_display(event['message'])}")
            if event.get("partial_count"):
                print(f"  保留已取得的 {event['partial_count']} 条候选，继续处理")
        elif kind == "news_cache_fallback":
            if event.get("scope_restricted") or event.get("date_restricted"):
                print("  本机缓存无法保证符合本次时间/来源条件，未采用缓存证据")
            else:
                print(f"  本机新闻缓存回退：{event['count']} 条（不是本次 API 返回）")
        elif kind == "news_filtered":
            print(f"  查询条件硬过滤：{event['input_count']} 条 → {event['matched_count']} 条")
            print(f"  精确时间：{event.get('published_after') or '不限'} ~ {event.get('published_before') or '不限'}")
            print(f"  信息源：{'、'.join(event.get('source_names', [])) or '不限'}；栏目：{event.get('section') or '不限'}")
        elif kind == "impact_progress":
            print(f"  [影响评分] {event['scored']}/{event['total']}；缓存复用 {event['cached']}")
        elif kind == "news_scope_progress":
            print(f"  [主题与影响筛选] 扫描 {event['scanned']}；相关 {event['accepted']}/{event['required']}；无关 {event['excluded']}")
        elif kind == "news_ranked":
            print(f"  匹配候选全部保留：{event['candidate_count']} 条；正文入选：{event['selected_count']} 条；暂未精读：{event.get('deferred_count', 0)} 条")
            if "raw_count" in event:
                print(f"  多组原始返回：{event['raw_count']} 条；去重并按查询条件过滤后：{event['candidate_count']} 条")
            print(f"  分页：{'已取完' if event.get('retrieval_complete') else '未取完/未知'}；评分：{event.get('ranking_method', '未记录')}（阅读优先级，不是热度）")
            print(f"  向量缓存：{'已使用' if event['vector_cache_used'] else '未使用；仍对全部候选做词法评分'}")
            if event.get("cache_stats") is not None:
                print(f"  向量缓存更新：{event['cache_stats']}")
        elif kind == "news_selected":
            print(f"  正文申请：{event.get('article_id')} / {event.get('title')}")
            print(f"  权重明细：{event.get('priority', {})}；原因：{event.get('selection_reason', '')}")
        elif kind == "news_article_error":
            print(f"  文章 {event['article_id']} 正文读取失败，使用已有摘要：{_display(event['message'])}")
        sys.stdout.flush()

    def show(self, node: str, update: dict[str, Any], before: dict[str, Any]) -> None:
        if self.compact is not None:
            return self.compact.show(node, update, before)
        if node == "initialize_research":
            self._heading("0", "建立可恢复研究")
            print(f"  研究编号：{update['run_id']}；工作任务上限：{settings.agent_workers}；模型请求上限：{settings.llm_concurrency}")
            return
        if node == "read_documents":
            self._heading("6b", "相关材料精读与成果持久化")
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
                estimate_label = {"none": "非预测数值", "directional": "方向", "level": "水平",
                                  "probability": "概率"}.get(update.get("estimate_kind"), update.get("estimate_kind"))
                if update.get("estimate_kind") and update.get("estimate_kind") != "none":
                    print(f"  估计类型：{estimate_label}；目标事件：{update.get('target_event') or '未单列'}；预测时点：{update.get('forecast_horizon') or '见原问题'}")
                print(f"  记忆实体词：{'、'.join(update.get('subject_terms', [])) or '未提供，仅作相关性筛选'}")
                print(f"  研究对象：{update.get('subject_scope') or '以问题语境为准'}")
            summary = _display(update.get("plan_summary"))
            if summary:
                print(f"  原因：{summary}")
            tool_status = {"required": "本轮必须调用", "conditional": "满足条件时调用", "skipped": "本轮跳过"}
            for decision in update.get("tool_plan", []):
                print(f"  工具计划/{decision['tool']}：{tool_status.get(decision['status'], decision['status'])}；"
                      f"调用者：{decision['caller']}；原因：{decision['reason']}")
            for source, query in update.get("source_queries", {}).items():
                print(f"  查询/{SOURCE_LABELS.get(source, source)}：{_display(query) or '（空：不按关键词筛选）'}")
            plan = update.get("news_search_plan", {})
            if plan:
                if plan.get("raw_queries") and plan["raw_queries"] != plan.get("queries"):
                    print(f"  模型原始新闻词：{plan['raw_queries']}")
                    print(f"  查询整理：拆分、去重和主题补全，最多 4 组；实际执行：{plan['queries']}")
                if plan.get("added_topic_queries"):
                    print(f"  主题补全查询：{plan['added_topic_queries']}")
                if plan.get("deferred_topics"):
                    print(f"  未单独检索主题：{plan['deferred_topics']}（本轮查询额度不足，未视为已覆盖）")
                suggestion = plan.get("suggested_time", {})
                mode = {"unrestricted": "不限时间", "suggested": "模型建议", "explicit": "用户指定"}.get(suggestion.get("mode"), "未提供")
                print(f"  模型时间选择：{mode}")
                print(f"  模型给出的日期：{_time_range(suggestion.get('start', ''), suggestion.get('end', ''))}")
                print(f"  模型给出的精确时间：{suggestion.get('published_after') or '不限'} ~ {suggestion.get('published_before') or '不限'}")
                print(f"  时间说明：{_display(suggestion.get('reason')) or '未提供'}")
                actual_time = "不执行查询（日期无效）" if plan.get("error") else _time_range(plan.get('start', ''), plan.get('end', ''))
                print(f"  实际采用日期：{actual_time}")
                print(f"  实际采用精确时间：{plan.get('published_after') or '不限'} ~ {plan.get('published_before') or '不限'}")
                print(f"  人物：{'、'.join(plan.get('people', [])) or '不限'}；机构：{'、'.join(plan.get('organizations', [])) or '不限'}")
                print(f"  主题：{'、'.join(plan.get('topics', [])) or '不限'}；信息源：{'、'.join(plan.get('source_names', [])) or '不限'}；栏目：{plan.get('section') or '不限'}")
                print(f"  排序：{plan.get('sort_by', 'relevance')}；覆盖：{plan.get('coverage', 'focused')}；入选上限：{plan.get('result_limit', settings.news_retrieval_k)}")
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
            print("  决策：相关或待确认材料进入精读，再做整批证据检查" if before.get("reading_recipe") else "  决策：进入整批证据检查")
            return

        if node == "assess_evidence":
            self._heading("6c", "检查证据是否足以分析")
            review = update.get("evidence_assessment", {})
            print(f"  可进行条件分析：{'是' if review.get('ready') else '证据仍有缺口'}")
            print(f"  判断：{_display(review.get('summary'))}")
            for factor in review.get("covered_factors", []):
                print(f"  已覆盖：{_display(factor)}")
            for factor in review.get("missing_factors", []):
                print(f"  缺口：{_display(factor)}")
            for concern in review.get("concerns", []):
                print(f"  必须回应 [{concern['id']}]：{_display(concern['detail'])}")
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
            for check in update.get("answer_review", {}).get("concern_checks", []):
                print(f"  [{check['concern_id']}] {'已回应' if check['addressed'] else '未解决'}：{_display(check['explanation'])}")
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


def run_with_trace(graph, question: str, *, verbose: bool = False, run_id=None, resume=False, conversation_input=None) -> dict[str, Any]:
    """流式运行图并合并节点更新，返回与 graph.invoke 相同用途的最终状态。"""
    state: dict[str, Any] = {**(conversation_input or {}), "question": question}
    trace = TracePrinter(verbose=verbose)
    input_state = dict(state)
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
            if not has_pending_work(snapshot):
                if recoverable_reading(snapshot):
                    raise ValueError("此记录有未读完文章，需要由 --resume 入口恢复阅读队列，不能作为最终答案展示")
                if not state.get("generation"):
                    raise ValueError("检查点没有最终答案，也没有可推进任务；不能判定为已完成")
                print("  该研究已结束，展示保存的结果（完成状态以核验标记为准）。")
                return state
            if state.get("model_revision"):
                from .research.service import model_revision, recipe, WORKFLOW_VERSION
                if (state["model_revision"] != model_revision()
                        or state.get("reading_recipe") != recipe(state["model_revision"])
                        or state.get("workflow_revision") != WORKFLOW_VERSION):
                    raise ValueError("模型、配置或工作流版本已变化，请发起新研究，避免混用旧检查点")
            input_state = None
    for mode, event in graph.stream(input_state, stream_mode=["updates", "custom"], **options):
        ledger = active_usage.get()
        if ledger is not None:
            ledger.drain(print_usage_event if verbose else print_compact_usage_event)
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
    from .news_api import NewsAPIError
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, NewsAPIError):
            return str(current)
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
        if name in {"ModelOutputTruncatedError", "ModelOutputValidationError", "ModelAdaptiveRetryError"}:
            return str(current)
        current = current.__cause__ or current.__context__
    return f"智能体执行失败：{type(exc).__name__}: {_clip(exc, 160)}"


def show_failure_context(database, run_id, exc, *, verbose=False):
    """诊断失败不能覆盖原始错误；保存类型和阶段，不保存隐藏思维链。"""
    message = "收到中断信号；日志不能判断是Ctrl+C、终端停止还是外部信号。" if isinstance(exc, KeyboardInterrupt) else _run_error_message(exc)
    try:
        database.event(run_id, {"kind": "run_failure", "error_type": type(exc).__name__, "message": message})
        report = inspect_research(database, run_id)
        print_inspection(report, include_usage=False)
        if verbose:
            print_inspection(report, verbose=True)
    except Exception as diagnostic_error:
        print(f"[诊断读取失败] {type(diagnostic_error).__name__}；原错误：{message}")


def ask(graph, question: str, *, verbose: bool = False, run_id=None, resume=False, retry_failed=False, session_ledger=None, conversation_input=None, on_result=None) -> bool:
    if not resume:
        print(f"\n问题：{question}")
    database = None
    acquired = False
    ledger = UsageLedger(observer=session_ledger.observe if session_ledger else None)
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
                    print(f"\n问题：{question}")
                    repair_reading = recoverable_reading(snapshot)
                    if snapshot.values.get("model_revision") and (has_pending_work(snapshot) or retry_failed or repair_reading):
                        from .research.service import model_revision, recipe, WORKFLOW_VERSION
                        if (snapshot.values["model_revision"] != model_revision()
                                or snapshot.values.get("reading_recipe") != recipe(snapshot.values["model_revision"])
                                or (snapshot.values.get("workflow_revision") != WORKFLOW_VERSION and not repair_reading)):
                            raise ValueError("模型、配置或工作流版本已变化，请发起新研究；兼容的阅读成果仍可复用")
                stack.enter_context(run_lease(database, run_id, question))
                acquired = True
                if resume and repair_reading:
                    from .research.service import WORKFLOW_VERSION
                    # 不重搜、不改原文、不清空已完成任务；仅重建缺失分段的队列。
                    graph.update_state({"configurable": {"thread_id": run_id}}, {
                        "documents": snapshot.values["research_documents"], "workflow_revision": WORKFLOW_VERSION,
                        "execution_violations": [], "generation": "", "generation_complete": False,
                        "generation_grounded": False, "specialist_findings": [], "specialist_execution": {},
                    }, as_node="grade_documents")
                    database.event(run_id, {"kind": "reading_recovery", "from": snapshot.values.get("workflow_revision"),
                                           "to": WORKFLOW_VERSION, "reason": "重建旧流程遗漏的阅读任务；复用已有成果"})
                    snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
                    print("已恢复未完成阅读：使用保存的原文，完成的分段直接复用。")
                if retry_failed:
                    count = database.retry_failed(run_id)
                    if count:
                        at_reading = "read_documents" in snapshot.next
                        graph.update_state({"configurable": {"thread_id": run_id}},
                            {"documents": snapshot.values.get("documents", []) if at_reading else
                                snapshot.values.get("research_documents", snapshot.values.get("documents", []))},
                            as_node="grade_documents" if at_reading else "collect_sources")
                        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
                    print(f"已重置 {count} 个失败任务的重试预算")
                database.run_status(run_id, "running")
                print(f"  研究编号：{run_id}；中断后用 agentic-rag --resume {run_id} 恢复", flush=True)
                history = database.token_events(run_id)
                ledger = UsageLedger(history=history, persist=lambda event: database.event(run_id, event),
                                     historical_gap=resume and not history,
                                     observer=session_ledger.observe if session_ledger else None)
            stack.enter_context(usage_session(ledger))
            from .budget import research_budget
            # 已完成历史仅展示，不受新的执行额度拦截；恢复按已计量执行时间累计，不算离线等待。
            offline = resume and not has_pending_work(snapshot) and not retry_failed
            if not offline:
                stack.enter_context(research_budget(
                    seconds=settings.research_total_timeout, calls=settings.research_max_model_calls,
                    elapsed=sum(s.get("elapsed") or 0 for s in ledger.sessions.values()
                                if s.get("event") == "session_finished"),
                    used_calls=len(ledger.calls)))
                print(f"  研究预算：累计执行 {settings.research_total_timeout or '不限'} 秒；模型调用 {settings.research_max_model_calls or '不限'} 次（含重试，已用 {len(ledger.calls)} 次）。")
                if database:
                    database.event(run_id, {"kind": "research_budget", "seconds": settings.research_total_timeout,
                        "max_model_calls": settings.research_max_model_calls, "used_calls": len(ledger.calls),
                        "historical_gap": ledger.historical_gap})
            extra = {"conversation_input": conversation_input} if conversation_input is not None else {}
            result = run_with_trace(graph, question, verbose=verbose, run_id=run_id, resume=resume, **extra)
            if database:
                database.run_status(run_id, "completed" if result.get("generation_complete") and result.get("generation_grounded") else "needs_attention")
    except KeyboardInterrupt as exc:
        if run_id and database and acquired:
            database.run_status(run_id, "interrupted")
            print(f"\n研究已中断，已完成成果保留。恢复：agentic-rag --resume {run_id}")
            show_failure_context(database, run_id, exc, verbose=verbose)
        else:
            print("\n[中断] KeyboardInterrupt；已记录用量保留。")
        print_usage_summary(ledger, session_ledger=session_ledger, compact=not verbose)
        return False
    except Exception as exc:
        if run_id and database and acquired:
            from .research.scheduler import ResearchWorkPending
            database.run_status(run_id, "needs_attention" if isinstance(exc, ResearchWorkPending) else "failed")
        print("\n[本次执行未完成]")
        print(f"  {_run_error_message(exc)}")
        print("  交互模式仍可继续，请直接重新输入问题。")
        if run_id and database and acquired:
            print(f"  恢复本次研究：agentic-rag --resume {run_id}（失败任务另加 --retry-failed）")
            show_failure_context(database, run_id, exc, verbose=verbose)
        if verbose:
            print(f"  异常类型：{type(exc).__module__}.{type(exc).__name__}")
        print_usage_summary(ledger, session_ledger=session_ledger, compact=not verbose)
        return False
    print_answer(result, ledger, session_ledger=session_ledger, verbose=verbose)
    if on_result is not None:
        try:
            on_result(result)
        except Exception as exc:
            # 记忆写入失败不重跑昂贵研究，也不能声称下一轮能看到它。
            print(f"[会话未保存] {type(exc).__name__}: {exc}；研究结果仍已保留。")
            return False
    # 正常走到END不等于完成用户任务；脚本调用也必须能识别needs_attention。
    return bool(result.get("generation_complete") and result.get("generation_grounded"))


def converse(graph, question, conversation, *, verbose=False, session_ledger=None):
    """会话理解先于研究：不确定就澄清；研究继续使用全新的 run/thread ID。"""
    ledger = UsageLedger(observer=session_ledger.observe if session_ledger else None)
    try:
        with usage_session(ledger, scope="conversation"):
            prepared = conversation.prepare(question)
    except KeyboardInterrupt:
        print("[会话理解已中断] 没有启动研究或写入半轮对话。")
        print_usage_summary(ledger, compact=not verbose)
        return False
    except Exception as exc:
        print(f"[会话读取失败] {type(exc).__name__}: {exc}；未启动研究。")
        print_usage_summary(ledger, compact=not verbose)
        return False
    decision = prepared["conversation_resolution"]
    decision["usage"] = ledger.report()["current"]
    print(f"[会话 {conversation.id}] 第 {prepared['conversation_revision'] + 1} 轮 | "
          f"摘要 {decision.get('compressed_turns', 0)} 轮 / 未装入 {decision.get('omitted_turns', 0)} 轮 | "
          f"上下文 {decision.get('context_bytes', 0)} UTF-8 字节（非精确 Token）")
    if ledger.report()["current"]["calls"]:
        print_usage_summary(ledger, compact=not verbose)
    run_id = uuid.uuid4().hex
    if decision["mode"] == "clarify":
        print(f"[需要澄清] {decision['clarification']}")
        try:
            conversation.remember(prepared, {"generation": decision["clarification"]}, run_id)
        except Exception as exc:
            print(f"[会话未保存] {type(exc).__name__}: {exc}")
        return False
    if decision["mode"] == "followup":
        print(f"[追问还原] {prepared['question']}")
    return ask(graph, prepared["question"], verbose=verbose, run_id=run_id, session_ledger=session_ledger,
               conversation_input=prepared,
               on_result=lambda result: conversation.remember(prepared, result, run_id))


def print_answer(result, ledger, *, session_ledger=None, verbose=False):
    """运行完成与离线查看共用同一答案/用量展示，不重新执行研究。"""
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
    print_usage_summary(ledger, session_ledger=session_ledger, compact=not verbose)


def main() -> None:
    # VS Code 终端使用 UTF-8；显式设置可避免 Windows 重定向输出时出现乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Multi-source agentic RAG")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--resume", metavar="RUN_ID", help="恢复已保存研究；不与新问题同时使用")
    parser.add_argument("--conversation", metavar="ID", help="接续指定会话（与 --resume 研究恢复不同）；省略时交互模式自动新建")
    parser.add_argument("--no-conversation", action="store_true", help="禁用会话记忆，每题独立执行")
    parser.add_argument("--runs", action="store_true", help="列出最近研究，不调用模型")
    parser.add_argument("--status", metavar="RUN_ID", help="查看任务与最近执行事件")
    parser.add_argument("--retry-failed", action="store_true", help="与 --resume 配合，重新尝试失败子任务")
    parser.add_argument("--inspect-reading", metavar="READING_ID", help="查看阅读成果、原文位置与版本")
    parser.add_argument("--repair-memory", action="store_true", help="仅重试向量投影，不重新阅读")
    parser.add_argument("--list-tools", action="store_true", help="查看实际注册的工具及参数，不调用模型或初始化数据库")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="交互模式等待新输入的最长秒数；默认 300，设为 0 可禁用",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="显示完整规划、证据核验、逐调用统计与诊断；默认只显示核心数字、参数和异常",
    )
    args = parser.parse_args()
    if args.resume and args.question:
        parser.error("--resume 不能同时提交新问题")
    if args.conversation and (args.resume or args.no_conversation):
        parser.error("--conversation 不能与 --resume 或 --no-conversation 同时使用")
    if args.conversation:
        from .conversation import validate_id
        try:
            validate_id(args.conversation)
        except ValueError as exc:
            parser.error(str(exc))
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed 需要 --resume")
    if args.idle_timeout < 0:
        parser.error("--idle-timeout 不能小于 0")
    if args.list_tools:
        from .tools import GUARDRAIL_TOOLS, TOOL_POLICIES, TOOLS
        print(json.dumps([{"name": item.name,
                           "category": "evidence" if item in TOOLS else "guardrail",
                           "caller": TOOL_POLICIES[item.name]["caller"],
                           "use_when": TOOL_POLICIES[item.name]["use_when"],
                           "description": item.description,
                           "input_schema": item.tool_call_schema.model_json_schema()}
                          for item in (*TOOLS, *GUARDRAIL_TOOLS)],
                         ensure_ascii=False, indent=2))
        return
    from .research.service import store
    database = store()
    if args.runs or args.status or args.inspect_reading or args.repair_memory:
        if args.runs:
            print(json.dumps(database.runs(), ensure_ascii=False, indent=2))
        if args.status:
            try:
                print_inspection(inspect_research(database, args.status), verbose=args.verbose)
            except ValueError as exc:
                print(f"[状态查询失败] {exc}")
                raise SystemExit(1)
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

    # 已结束检查点是可离线展示的成果，不依赖 Ollama、索引监听或当前模型配置。
    # 只读快照后直接展示，不进入 ask()，避免并发状态变化触发意外重跑。
    if args.resume and not args.retry_failed:
        from .research.inspection import read_snapshot
        try:
            snapshot = read_snapshot(args.resume)
        except (OSError, sqlite3.Error):
            snapshot = None
        if snapshot and snapshot.values.get("generation") and not has_pending_work(snapshot) and not recoverable_reading(snapshot):
            print(f"\n问题：{snapshot.values.get('question', '')}\n研究编号：{args.resume}")
            print("该研究已结束，展示保存的结果；不连接模型、不刷新新闻、不改写研究记录。")
            from .research.service import WORKFLOW_VERSION
            if snapshot.values.get("workflow_revision") != WORKFLOW_VERSION:
                print("[历史版本提示] 此结果未经过当前交付约束与证据检查；旧的通过标记不代表已通过新规则。要重新验证请发起新研究。")
            history = database.token_events(args.resume)
            print_answer(dict(snapshot.values), UsageLedger(history=history, historical_gap=not history),
                         verbose=args.verbose)
            return

    # 终端使用结构化步骤追踪；第三方库只保留错误，避免 HTTP 和回退警告淹没主流程。
    logging.basicConfig(level=logging.ERROR, format="  %(message)s")

    watcher = None
    model_cleanup_needed = False
    session_ledger = UsageLedger()
    print(f"模型思考：{'开启' if settings.llm_reasoning else '关闭'}；生成额度（含思考）：{settings.llm_max_output_tokens} token")
    try:
        if settings.knowledge_watch_enabled:
            watcher = start_knowledge_watcher()

        if settings.ollama_warmup_enabled:
            if not warm_up_model(session_ledger=session_ledger, verbose=args.verbose):
                print("Ollama 模型未就绪，程序停止；请检查 Ollama 后重新启动。")
                raise SystemExit(1)

        model_cleanup_needed = True

        from langgraph.checkpoint.sqlite import SqliteSaver
        Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as resources:
            saver = resources.enter_context(SqliteSaver.from_conn_string(settings.checkpoint_db))
            graph = build_graph(checkpointer=saver)

            if args.resume:
                succeeded = ask(graph, "", verbose=args.verbose, run_id=args.resume, resume=True, retry_failed=args.retry_failed, session_ledger=session_ledger)
                if not succeeded:
                    raise SystemExit(1)
                return

            if args.question:
                if args.conversation:
                    from .conversation import Conversation, ConversationStore
                    conversation = Conversation(ConversationStore(settings.conversation_db), args.conversation)
                    succeeded = converse(graph, " ".join(args.question), conversation, verbose=args.verbose, session_ledger=session_ledger)
                else:
                    succeeded = ask(graph, " ".join(args.question), verbose=args.verbose, run_id=uuid.uuid4().hex, session_ledger=session_ledger)
                if not succeeded:
                    raise SystemExit(1)
                return

            timeout_note = (
                f"；空闲 {args.idle_timeout:g} 秒自动退出"
                if args.idle_timeout > 0 else ""
            )
            print(f"Agentic RAG — 交互模式（Ctrl+C 或输入 exit 退出{timeout_note}）")
            conversation = None
            if not args.no_conversation:
                from .conversation import Conversation, ConversationStore
                conversation = Conversation(ConversationStore(settings.conversation_db), args.conversation)
                print(f"会话编号：{conversation.id}；/new 新会话；/history 最近记录；/turn N 回读原文。")
            while True:
                try:
                    question = input_with_timeout("\n> ", args.idle_timeout).strip()
                except ConsoleInputTimeout:
                    print(f"\n超过 {args.idle_timeout:g} 秒没有新输入，自动退出。")
                    break
                except (KeyboardInterrupt, EOFError):
                    break
                if not question or question.lower() in {"exit", "quit"}:
                    break
                if conversation is not None:
                    if question == "/new":
                        conversation = Conversation(conversation.database)
                        print(f"已切换新会话：{conversation.id}（旧会话保留，不再注入）")
                        continue
                    if question == "/history":
                        _, turns = conversation.database.snapshot(conversation.id, limit=20)
                        for turn in turns:
                            print(f"{turn['sequence']}. [{turn['status']}] {_clip(turn['question'], 120)} | 研究 {turn['run_id']}")
                        continue
                    if question.startswith("/turn "):
                        try:
                            turn = conversation.database.turn(conversation.id, int(question[6:].strip()))
                            print(json.dumps(turn, ensure_ascii=False, indent=2))
                        except ValueError as exc:
                            print(f"[回读失败] {exc}")
                        continue
                    converse(graph, question, conversation, verbose=args.verbose, session_ledger=session_ledger)
                else:
                    ask(graph, question, verbose=args.verbose, run_id=uuid.uuid4().hex, session_ledger=session_ledger)
    finally:
        if watcher is not None:
            watcher.stop()
            watcher.join()
        if model_cleanup_needed:
            unload_model(verbose=args.verbose)


if __name__ == "__main__":
    main()
