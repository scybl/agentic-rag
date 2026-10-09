"""将用户的交付要求落实为可检查的条件，而不是只交给反思模型判断。"""

import re


def _number(value):
    if value.isdigit():
        return int(value)
    digits = dict(zip("零一二三四五六七八九", range(10)))
    value = value.replace("两", "二")
    if "百" in value:
        head, tail = value.split("百", 1)
        return digits.get(head, 1) * 100 + (_number(tail.lstrip("零")) if tail.lstrip("零") else 0)
    if "十" in value:
        head, tail = value.split("十", 1)
        return digits.get(head, 1) * 10 + digits.get(tail, 0)
    return digits.get(value, 0)


def parse_contract(question):
    """仅提取明确数量，不凭空添加日期窗口；阿拉伯数字和常用中文数字均可。"""
    quantities = list(re.finditer(r"(\d+|[一二两三四五六七八九十百]+)\s*[篇条则]\s*(?:新闻|报道|资讯)?", question))
    if not quantities or not re.search(r"新闻|报道|资讯", question):
        return {}
    first = _number(quantities[0][1])
    selected = _number(quantities[-1][1]) if len(quantities) > 1 else first
    if not 1 <= selected <= first <= 200 or selected > 100:
        raise ValueError("当前支持候选 1–200 篇、分析 1–100 篇，分析数不能超过候选数；不会擅自缩减数量。")
    return {"candidate_limit": first, "selection_limit": selected,
            "latest": bool(re.search(r"最近|最新|近期|latest|recent", question, re.I)),
            "ranking_mode": "impact" if re.search(r"影响.{0,8}(?:最大|重要|排序)|最.{0,4}影响|重要性|最重要", question) else "relevance"}


def apply_contract(plan, contract):
    if not contract:
        return plan
    plan = {**plan, **contract, "result_limit": contract["selection_limit"]}
    # 先确定用户指定主题的集合，不能让宏观扩展词改变“最新 N 篇”的母集。
    if contract["latest"]:
        from .news_plan import query_covers_term, validate_api_query
        primary = plan.get("query", "").strip()
        if not primary:
            raise ValueError("指定最新新闻数量时必须生成明确的主题查询，不能以全库替代。")
        required_entities = [*plan.get("people", []), *plan.get("organizations", [])]
        primary = " ".join([primary, *(term for term in required_entities if not query_covers_term(primary, term))]).strip()
        validate_api_query(primary, plan.get("section", ""))
        plan["supporting_queries"] = [q for q in plan.get("queries", []) if q != primary]
        plan["queries"] = [primary]
        plan["sort_by"] = "newest"
        if plan.get("suggested_time", {}).get("mode") != "explicit":
            for key in ("start", "end", "published_after", "published_before"):
                plan[key] = ""
            plan["time_note"] = "按用户要求取最新数量，不自动添加日期窗口；模型建议保留供查看"
    return plan


def completion_violations(state, *, final=True):
    """硬约束不会被模型的 accept 覆盖；不满足时保留成果，但不能宣布完成。"""
    problems = []
    reports = state.get("reading_reports", [])
    partial = [r for r in reports if r.get("status") != "complete" or r.get("covered", 0) != r.get("total", 0)]
    if partial:
        problems.append(f"仍有 {len(partial)} 篇新闻未完成分段阅读")
    dispatch = state.get("specialist_execution", {})
    # 重新进入证据检查时，旧专题失败结果不能阻止调度器重试；最终交付仍必须全部完成。
    if final and dispatch and dispatch.get("completed", 0) < dispatch.get("expected", 0):
        problems.append(f"专题分析仅完成 {dispatch.get('completed', 0)}/{dispatch['expected']}")
    contract = state.get("task_contract", {})
    if not contract:
        return problems
    batches = [e for e in state.get("news_trace", []) if e.get("kind") == "news_ranked"]
    first = batches[0] if batches else {}
    if first.get("candidate_count") != contract["candidate_limit"] or not first.get("retrieval_complete"):
        problems.append(f"未确认取得要求的 {contract['candidate_limit']} 篇候选新闻")
    if contract["latest"] and first.get("selection_scope") != "latest_n":
        problems.append("没有按最新 N 篇建立候选集合")
    if contract["latest"] and not first.get("subject_scope_checked"):
        problems.append("最新候选未核对主题关系，同名词误命中不能算作相关新闻")
    if contract["ranking_mode"] == "impact" and first.get("ranking_method") != "impact-v1":
        problems.append("影响力评分未完成，不能用相似度替代")
    required = contract["selection_limit"]
    news = [d for d in state.get("answer_documents", []) if d.metadata.get("source_type") == "news_api"]
    unique = {d.metadata.get("article_id") for d in news if d.metadata.get("article_id")}
    if len(unique) != required:
        problems.append(f"最终分析新闻为 {len(unique)}/{required} 篇")
    selected = {e.get("article_id") for e in state.get("news_trace", []) if e.get("kind") == "news_selected"}
    if selected and unique != selected:
        problems.append("交付清单与本轮入选文章不一致，不能用集合外材料补数量")
    if final and state.get("task_type") != "factual" and not dispatch:
        problems.append("没有可核对的专题任务执行结果")
    if any(d.metadata.get("content_kind") != "news_article" for d in news):
        problems.append("部分入选新闻只有摘要，不能视为完成原文分析")
    if "完整事实块超过上下文预算" in state.get("evidence_context", ""):
        problems.append("部分入选新闻没有完整事实块进入生成上下文，需增加上下文预算或分层综合")
    if not state.get("reading_recipe") or len([r for r in reports if r.get("status") == "complete"]) < required:
        problems.append(f"未完整阅读要求的 {required} 篇新闻")
    return list(dict.fromkeys(problems))


def selection_appendix(state):
    """从已保存的评分与阅读成果生成可审计清单，避免模型遗漏或杜撰名单。"""
    if not state.get("task_contract"):
        return ""
    rows = []
    def cell(value):
        return str(value).replace("|", "／").replace("\n", " ")
    for i, doc in enumerate(state.get("answer_documents", []), 1):
        if doc.metadata.get("source_type") != "news_api":
            continue
        impact = doc.metadata.get("impact", {})
        mechanism = impact.get("mechanism", "按相关性入选")
        # 加息概率下降不是降息预期；历史评分也不能在展示时传播这个术语替换错误。
        support = str(impact.get("quote", ""))
        if "降息" in mechanism and "降息" not in support and re.search(r"加息|加息概率", support) and re.search(r"下调|下降|降温|削减", support):
            mechanism = "来源仅表明加息预期减弱，不代表降息预期；潜在影响仍需核对实际利率与美元。"
        claims = doc.metadata.get("reading_claims", [])
        fact = next((c for c in claims if c["kind"] == "reported_fact"), claims[0] if claims else {})
        background = fact.get("event_context", "")
        if background.startswith(("原文未明确事件时间", "未取得独立时间背景")):
            background = "时间以引文为准；不按发布日期推断。"
        statement = fact.get("statement", "未取得阅读成果")
        if fact.get("kind") == "source_excerpt":
            from .evidence import whole_sentence_excerpt
            # 短文全量持久化不等于把整篇原文塞进终端表格；只展示有界的完整原句。
            statement = whole_sentence_excerpt(statement, state.get("question", ""), 180)
        rows.append(f"| {len(rows)+1} | {cell(doc.metadata.get('title', ''))} [E{i}] | "
                    f"{cell(doc.metadata.get('published_at', ''))[:10]} | {impact.get('score', '—')} | "
                    f"{cell(mechanism)} | "
                    f"{cell(background)} "
                    f"【{ {'reported_fact': '来源报道', 'attributed_forecast': '来源预测', 'opinion': '观点', 'method': '方法', 'source_excerpt': '原文节选，未独立核实'}.get(fact.get('kind'), '未确认')}】"
                    f"{cell(statement)} |")
    if not rows:
        return ""
    return ("\n\n### 入选新闻逐篇分析\n\n影响分是基于候选摘要的潜在影响评估，不是已实证的价格贡献；原文阅读事实另列。\n\n"
            "| 序号 | 新闻 | 发布时间 | 影响分 / 100 | 潜在传导与方向 | 原文阅读要点 |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows))
