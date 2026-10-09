"""受控工具调用入口：真实 ToolCall/ToolMessage、参数校验与观察事件。"""

import time
import uuid
from dataclasses import replace

from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import ensure_config, merge_configs
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from .contracts import EvidenceBundle, ModelRetryPlan, RuleBundle, ToolContext


def _audit_value(value, *, depth=0):
    """事件只展示有界参数；完整的已校验参数仍交给工具。"""
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:239] + "…"
    if isinstance(value, list):
        items = [_audit_value(item, depth=depth + 1) for item in value[:8]]
        return items + ([f"…另有 {len(value) - 8} 项"] if len(value) > 8 else [])
    if isinstance(value, dict) and depth < 3:
        return {str(key): _audit_value(item, depth=depth + 1) for key, item in value.items()}
    return value


def execute_tool(tool: BaseTool, arguments: dict, *, context: ToolContext | None = None) -> ToolMessage:
    """只执行一次；重试和流程跳转由现有运行控制层负责。

    编排器创建调用编号，不假装这些调用来自模型原生 tool_calls。
    真实 ToolMessage 保留相同编号及完整 artifact，可供现有图或未来工具循环复用。
    """
    context = context or ToolContext()
    caller = context.caller.strip()
    reason = context.reason.strip()
    if not caller or not reason:
        raise ValueError("工具调用必须声明 caller 和 reason，才能进入统一审计入口")
    call_id = uuid.uuid4().hex
    started = time.monotonic()

    def emit(event):
        if event.get("kind") == "tool" and event.get("tool_call_id") and event["tool_call_id"] != call_id:
            # 嵌套模型校验/重试必须保留自己的工具身份，不能冒充外层新闻搜索。
            context.emit({"parent_tool_call_id": call_id, **event})
            return
        context.emit({**event, "tool": tool.name, "tool_call_id": call_id,
                      "caller": caller, "reason": event.get("reason", reason), "tool_reason": reason})

    try:
        # 在写日志之前拒绝额外参数；只展示工具白名单中可公开的查询字段。
        normalized = tool.args_schema.model_validate(arguments).model_dump()
        emit({"kind": "tool", "phase": "started", "arguments": _audit_value(normalized)})
        message = tool.invoke(
            {"name": tool.name, "args": normalized, "id": call_id, "type": "tool_call"},
            # 保留图的流写入器、检查点命名空间和回调，只增补工具上下文。
            config=merge_configs(ensure_config(), {"configurable": {"tool_context": replace(context, emit=emit)}}),
        )
        if not isinstance(message, ToolMessage) or not isinstance(
            message.artifact, (EvidenceBundle, RuleBundle, ModelRetryPlan)
        ):
            raise TypeError(f"工具 {tool.name} 未返回约定的 artifact")
        bundle = message.artifact
        message.status = "error" if bundle.status == "error" else "success"
        finished = {"kind": "tool", "phase": "finished", "status": bundle.status,
                    "count": bundle.count, "warnings": bundle.warnings,
                    "elapsed": round(time.monotonic() - started, 3)}
        if isinstance(bundle, RuleBundle):
            finished.update(rule=bundle.rule, passed=bundle.passed, checks=bundle.checks)
        elif isinstance(bundle, ModelRetryPlan):
            finished.update(
                action=bundle.action, retry=bundle.retry, plan_reason=bundle.reason,
                adjustments={
                    "num_predict": bundle.next_num_predict,
                    "num_ctx": bundle.next_num_ctx,
                    "timeout_seconds": bundle.next_timeout_seconds,
                    "reasoning": bundle.next_reasoning,
                },
            )
        emit(finished)
        return message
    except Exception as exc:
        # ValidationError 默认字符串会包含原始输入，避免意外记录未声明的敏感值。
        error = "；".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()) if isinstance(exc, ValidationError) else str(exc)[:500]
        emit({"kind": "tool", "phase": "failed", "error_type": type(exc).__name__,
              "error": error, "elapsed": round(time.monotonic() - started, 3)})
        raise
