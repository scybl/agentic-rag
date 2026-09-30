"""显式工具目录；导入不会连接模型、访问网络或建立索引。

仅把模型可请求的取证能力暴露为工具，调度、缓存写入和反思不注册。
"""

from .knowledge import search_knowledge
from .news import search_news
from .news_reading import read_news
from .web_search import search_web
from .guardrails import (plan_model_retry, validate_model_output,
                         validate_probability_answer, validate_probability_evidence)


SOURCE_TOOLS = {
    "vectorstore": search_knowledge,
    "news_api": search_news,
    "web_search": search_web,
}
TOOLS = (*SOURCE_TOOLS.values(), read_news)
GUARDRAIL_TOOLS = (validate_model_output, plan_model_retry,
                   validate_probability_evidence, validate_probability_answer)

TOOL_POLICIES = {
    "search_knowledge": {
        "category": "evidence", "caller": "主流程/证据收集器",
        "use_when": "需要本地方法、领域概念或稳定知识；概率任务强制检索估算方法",
    },
    "search_news": {
        "category": "evidence", "caller": "主流程/证据收集器",
        "use_when": "需要当前或历史新闻事实；政策概率任务用于寻找带正文的市场定价报道",
    },
    "search_web": {
        "category": "evidence", "caller": "主流程/证据收集器",
        "use_when": "需要公开核验、原始数据线索或其他来源的缺口；概率任务强制使用",
    },
    "read_news": {
        "category": "evidence", "caller": "主流程或专题Agent",
        "use_when": "已有新闻版本但摘要/提取不足以核对关键数字、口径或方法时定向回读",
    },
    "validate_probability_evidence": {
        "category": "guardrail", "caller": "主流程/概率证据契约",
        "use_when": "概率任务生成前强制检查直接概率数据、方法和正文证据",
    },
    "validate_probability_answer": {
        "category": "guardrail", "caller": "主流程/概率答案契约",
        "use_when": "概率任务交付前强制检查事件、时点、数值、方法和引用",
    },
    "validate_model_output": {
        "category": "guardrail", "caller": "模型输出/结构校验器",
        "use_when": "每次模型返回后强制检查目标类型与字段结构；失败时反馈具体原因并要求重做",
    },
    "plan_model_retry": {
        "category": "guardrail", "caller": "模型调用/自适应控制器",
        "use_when": "模型输出截断、超时、连接/服务故障、空输出或结构错误后，按硬预算选择调整或停止",
    },
}


def plan_tool_usage(*, selected_sources: list[str], estimate_kind: str, research: bool) -> list[dict]:
    """为全部注册工具给出本轮的明确调用状态；未选中也必须说明原因。"""
    source_names = {source: tool.name for source, tool in SOURCE_TOOLS.items()}
    selected_names = {source_names[source] for source in selected_sources if source in source_names}
    decisions = []
    for tool in (*TOOLS, *GUARDRAIL_TOOLS):
        policy = TOOL_POLICIES[tool.name]
        if tool.name in selected_names:
            status, reason = "required", f"规划已选择对应数据源；{policy['use_when']}"
        elif tool.name == "read_news":
            if research and "search_news" in selected_names:
                status, reason = "conditional", policy["use_when"]
            else:
                status, reason = "skipped", "本轮没有可回读的持久化新闻证据"
        elif tool.name == "validate_model_output":
            status, reason = "required", policy["use_when"]
        elif tool.name == "plan_model_retry":
            status, reason = "conditional", policy["use_when"]
        elif tool.name in {"validate_probability_evidence", "validate_probability_answer"}:
            if estimate_kind == "probability":
                status, reason = "required", policy["use_when"]
            else:
                status, reason = "skipped", "本轮不是概率估计，概率专用契约不适用"
        else:
            status, reason = "skipped", "规划未选择该数据源，当前任务没有对应取证需要"
        decisions.append({"tool": tool.name, "status": status, "caller": policy["caller"], "reason": reason})
    return decisions


def describe_tools() -> str:
    """从实际工具定义生成规划提示，避免工具名字与实现各写一套。"""
    descriptions = []
    for source, item in SOURCE_TOOLS.items():
        inputs = ", ".join(item.tool_call_schema.model_json_schema()["properties"])
        descriptions.append(f"{source} -> {item.name}({inputs}): {item.description}")
    descriptions.append(f"专题回读 -> {read_news.name}(evidence_ids, goal): {read_news.description}")
    return "\n".join(descriptions)


__all__ = ["TOOLS", "GUARDRAIL_TOOLS", "TOOL_POLICIES", "SOURCE_TOOLS", "describe_tools", "plan_tool_usage",
           "search_knowledge", "search_news", "search_web", "read_news", "validate_model_output", "plan_model_retry",
           "validate_probability_evidence", "validate_probability_answer"]
