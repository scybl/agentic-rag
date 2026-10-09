"""精简终端视图：保留数字、实际参数和异常；长解释交给 -v。"""

from collections import defaultdict

from .token_usage import format_duration


SOURCES = {"vectorstore": "知识", "news_api": "新闻", "web_search": "网络"}


def one_line(value):
    return " ".join(str(value or "").split())


def dates(start="", end=""):
    return "不限" if not start and not end else f"{start or '不限'}~{end or '不限'}"


class CompactTrace:
    def __init__(self):
        self.queued = defaultdict(set)
        self.done = defaultdict(set)
        self.screened = defaultdict(int)

    def show_event(self, event):
        kind = event.get("kind")
        if kind == "research":
            name, role = event.get("event"), event.get("role", "任务")
            task = event.get("task_key") or event.get("task_id", "")
            if name == "queued":
                self.queued[role].add(task)
            elif name in {"completed", "reused"}:
                self.done[role].add(task)
                status = "复用" if name == "reused" else "完成"
                print(f"[{role}] {status} {len(self.done[role])}/{len(self.queued[role]) or '?'} | "
                      f"任务 {task[:12]} | 耗时 {format_duration(event.get('elapsed'))}", flush=True)
            elif name in {"failed", "attempt_failed", "timeout", "budget", "blocked", "deferred"}:
                print(f"[任务异常] {name} | {task[:12]} | {one_line(event.get('error'))}", flush=True)
            elif name == "reading_split":
                print(f"[阅读缩段] {event.get('characters')} 字符 | 最多二分一次", flush=True)
            elif name == "verbatim_read":
                print(f"[短文原样保留] {event['characters']} 字符 | 阅读模型调用 0 | 未独立核实", flush=True)
            elif name == "budget_plan":
                print(f"[任务计划] 需要 {event['required']} / 数量上限 {event['allowed'] or '不限'} | 专题预留 {event['reserved_specialists']} | 并发 {event.get('workers', '?')}", flush=True)
            elif name == "reread_context_limit":
                print(f"[回读上下文] 已纳入 {event['included']}/{event['total']} 块 | {event['reason']}", flush=True)
            elif name == "specialist_advice_filtered":
                print(f"[专题建议校验] 未采纳含未核对数值的句子 | {event['reason']}", flush=True)
            elif name == "index_sync" and event.get("failed"):
                print(f"[向量写入] 成功 {event['synced']} / 失败 {event['failed']} / 待处理 {event['remaining']}", flush=True)
        elif kind == "tool":
            if event.get("phase") == "started":
                print(f"[工具调用] {event.get('tool')} | 调用者 {one_line(event.get('caller'))} | "
                      f"原因 {one_line(event.get('reason'))}", flush=True)
            elif event.get("phase") == "finished":
                labels = {"ok": "成功", "empty": "零结果", "degraded": "降级",
                          "error": "失败", "rejected": "规则未通过",
                          "adjusted": "动态调整", "redirected": "改用其他策略",
                          "stopped": "停止重试"}
                print(f"[工具结果] {event.get('tool')} | {labels.get(event.get('status'), event.get('status'))} | "
                      f"项目 {event.get('count', 0)} | {format_duration(event.get('elapsed'))}", flush=True)
                if event.get("action"):
                    changes = event.get("adjustments", {})
                    print(f"[模型调整] {event['action']} | 输出 {changes.get('num_predict')} | "
                          f"上下文 {changes.get('num_ctx')} | 超时 {changes.get('timeout_seconds')}s | "
                          f"思考 {'开' if changes.get('reasoning') else '关'} | "
                          f"{one_line(event.get('plan_reason'))}", flush=True)
                for warning in event.get("warnings", []):
                    print(f"[工具提示] {event.get('tool')} | {one_line(warning)}", flush=True)
            elif event.get("phase") == "failed":
                print(f"[工具异常] {event.get('tool')} | {event.get('error_type', '')} | "
                      f"{one_line(event.get('error'))}", flush=True)
        elif kind == "news_retry":
            print(f"[新闻重试] HTTP {event['status_code']} | 第 {event['attempt']}/{event['max_attempts']} 次 | "
                  f"等待 {event['delay']} 秒 | q={event.get('query')!r}", flush=True)
        elif kind == "news_request":
            print(f"[新闻请求] q={event.get('query')!r} | 日期 {dates(event.get('start'), event.get('end'))} | "
                  f"栏目 {event.get('section') or '不限'} | 第 {event['page']} 页 / 上限 {event['limit']}", flush=True)
        elif kind == "news_page":
            print(f"[新闻返回] 第 {event['page']} 页 {event['count']} 条 | 续页 {'有' if event['has_more'] else '无'}", flush=True)
        elif kind == "news_ranked":
            print(f"[新闻排序] 原始 {event.get('raw_count', event['candidate_count'])} → 匹配候选 {event['candidate_count']} | "
                  f"正文预算入选 {event['selected_count']} / 暂未精读 {event.get('deferred_count', 0)} | "
                  f"请求范围 {'已取完' if event.get('retrieval_complete') else '未取完/未知'} / {event.get('selection_scope', 'all_matches')} | "
                  f"评分 {event.get('ranking_method', '未记录')}（非热度）", flush=True)
        elif kind == "news_selected":
            priority = event.get("impact") or event.get("priority", {})
            print(f"[新闻精读选择] #{priority.get('rank', '?')} | 权重 {priority.get('score', '?')} | "
                  f"{one_line(event.get('title'))} | {event.get('article_id')} | "
                  f"{event.get('selection_reason', '')}", flush=True)
        elif kind == "impact_progress":
            print(f"[影响评分] {event['scored']}/{event['total']} | 复用 {event['cached']}", flush=True)
        elif kind in {"news_screen_progress", "news_screen_batch"}:
            stage = event["stage"]
            if kind == "news_screen_progress":
                self.screened[stage] = event["processed"]
                print(f"[{'摘要初筛' if stage == 'abstract' else '正文主题核对'}] {event['processed']}/{event['total']} | 复用 {event['cached']} | 每批 {event['batch_size']}", flush=True)
            else:
                self.screened[stage] += event["count"]
                print(f"[{'摘要初筛' if stage == 'abstract' else '正文主题核对'}] 已处理 {self.screened[stage]} | 本批明确无关 {event['excluded']}", flush=True)
        elif kind == "news_core_replaced":
            print(f"[核心文章补位] {event.get('article_id')} | {event['reason']}", flush=True)
        elif kind == "news_core_ready":
            print(f"[核心正文] {event['selected']}/{event['limit']} | 摘要 {event['abstracts']} / 初筛可用 {event['eligible']} | 正文请求 {event['attempted']}", flush=True)
        elif kind == "news_scope_progress":
            print(f"[主题与影响筛选] 已扫描 {event['scanned']} | 相关候选 {event['accepted']}/{event['required']} | 无关 {event['excluded']}", flush=True)
        elif kind == "news_filtered":
            print(f"[新闻过滤] {event['input_count']} → {event['matched_count']} | "
                  f"精确时间 {event.get('published_after') or '不限'} ~ {event.get('published_before') or '不限'} | "
                  f"来源 {'、'.join(event.get('source_names', [])) or '不限'}", flush=True)
        elif kind in {"news_error", "news_article_error"}:
            print(f"[新闻异常] {one_line(event.get('message'))} | 已保留 {event.get('partial_count', 0)} 条", flush=True)
        elif kind == "news_cache_fallback":
            print("[新闻回退] 时间/来源条件无法保证，未采用缓存" if event.get("scope_restricted") or event.get("date_restricted") else
                  f"[新闻回退] 本地缓存 {event['count']} 条，非本次API结果", flush=True)

    def show(self, node, update, before):
        if node == "route":
            print(f"[规划] 来源 {'+'.join(SOURCES.get(s, s) for s in update.get('selected_sources', []))} | "
                  f"因素 {len(update.get('evidence_needs', []))} | 类型 {update.get('task_type', '未标注')} | "
                  f"估计 {update.get('estimate_kind', 'none')}")
            labels = {"required": "必须", "conditional": "条件", "skipped": "跳过"}
            for decision in update.get("tool_plan", []):
                print(f"[工具计划/{labels.get(decision['status'], decision['status'])}] {decision['tool']} | "
                      f"调用者 {decision['caller']} | 原因 {one_line(decision['reason'])}")
            for source, query in update.get("source_queries", {}).items():
                print(f"[查询] {SOURCES.get(source, source)}={one_line(query)}")
            plan = update.get("news_search_plan", {})
            contract = update.get("task_contract", {})
            if contract:
                print(f"[交付约束] 候选 {contract['candidate_limit']} / 分析 {contract['selection_limit']} | "
                      f"最新优先 {contract['latest']} | 排名 {contract['ranking_mode']}")
                if plan.get("supporting_queries"):
                    print(f"[范围约束] 候选仅用主主题；扩展词未混入：{' / '.join(plan['supporting_queries'])}")
            if plan:
                if plan.get("added_topic_queries"):
                    print(f"[查询补全] {' / '.join(plan['added_topic_queries'])}")
                if plan.get("deferred_topics"):
                    print(f"[主题未单独检索] {'、'.join(plan['deferred_topics'])} | 本轮最多4组查询，未视为已覆盖")
                suggestion = plan.get("suggested_time", {})
                print(f"[日期] 模型={dates(suggestion.get('start'), suggestion.get('end'))} | "
                      f"实际={dates(plan.get('start'), plan.get('end'))} | {plan.get('time_note', '')}")
                print(f"[新闻指令] 精确时间 {plan.get('published_after') or '不限'} ~ {plan.get('published_before') or '不限'} | "
                      f"人物 {'、'.join(plan.get('people', [])) or '不限'} | 机构 {'、'.join(plan.get('organizations', [])) or '不限'} | "
                      f"主题 {'、'.join(plan.get('topics', [])) or '不限'} | 来源 {'、'.join(plan.get('source_names', [])) or '不限'} | "
                      f"栏目 {plan.get('section') or '不限'} | 排序 {plan.get('sort_by', 'relevance')} | "
                      f"覆盖 {plan.get('coverage', 'focused')} | 入选 {plan.get('result_limit', 5)}")
                if plan.get("error"):
                    print(f"[日期错误] {plan['error']}")
        elif node in {"collect_sources", "supplement_sources"}:
            counts = " / ".join(f"{SOURCES.get(k, k)} {len(v)}" for k, v in update.get('documents_by_source', {}).items())
            print(f"[证据] {counts} | 合并 {len(update.get('documents', []))} | 补搜 {update.get('retries', 0)}")
            for source, error in update.get("source_errors", {}).items():
                print(f"[来源异常] {SOURCES.get(source, source)} | {one_line(error)}")
        elif node == "grade_documents":
            print(f"[筛选] 保留 {len(update.get('documents', []))}/{len(before.get('documents', []))}")
        elif node == "read_documents":
            reports = update.get("reading_reports", [])
            print(f"[阅读] {len(reports)} 篇 | 覆盖 {sum(r['covered'] for r in reports)}/{sum(r['total'] for r in reports)} 段")
        elif node == "assess_evidence":
            review = update.get("evidence_assessment", {})
            print(f"[证据检查] 可用 {len(review.get('usable_evidence_ids', []))} | 缺口 {len(review.get('missing_factors', []))} | "
                  f"风险 {len(review.get('concerns', []))} | 动作 {update.get('next_action')}")
            self.pending(update)
        elif node == "dispatch_specialists":
            findings = update.get("specialist_findings", [])
            if findings:
                print(f"[专题] 成果 {len(findings)} | 动作 {update.get('next_action')}")
            self.pending(update)
        elif node in {"generate", "revise_answer"}:
            print(f"[答案] 证据 {len(update.get('answer_documents', before.get('documents', [])))} | "
                  f"缓存 {'命中' if update.get('analysis_cache_hit') else '未命中'}")
        elif node == "evaluate_generation":
            review = update.get("answer_review", {})
            print(f"[核验] 依据 {'通过' if update.get('generation_grounded') else '未通过'} / "
                  f"任务 {'通过' if update.get('generation_complete') else '未完成'} | 问题 {len(review.get('issues', []))} | 动作 {update.get('next_action', 'finish')}")
            for issue in review.get("issues", []):
                print(f"[核验问题] {one_line(issue)}")
            self.pending(update)

    @staticmethod
    def pending(update):
        status = "待执行" if update.get("next_action") == "supplement" else "建议，未执行"
        for key, label in [("pending_news_queries", "新闻"), ("pending_web_queries", "网络")]:
            if update.get(key):
                print(f"[补搜/{status}] {label}={'; '.join(update[key])}")
