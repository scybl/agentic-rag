"""受控工具调用入口：真实 ToolCall/ToolMessage、参数校验与观察事件。"""

import time
import uuid
from dataclasses import replace

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from .contracts import EvidenceBundle, ToolContext


def execute_tool(tool: BaseTool, arguments: dict, *, context: ToolContext | None = None) -> ToolMessage:
    """只执行一次；重试和流程跳转由现有运行控制层负责。

    编排器创建调用编号，不假装这些调用来自模型原生 tool_calls。
    真实 ToolMessage 保留相同编号及完整 artifact，可供现有图或未来工具循环复用。
    """
    context = context or ToolContext()
    call_id = uuid.uuid4().hex
    started = time.monotonic()

    def emit(event):
        context.emit({**event, "tool": tool.name, "tool_call_id": call_id})

    try:
        # 在写日志之前拒绝额外参数；只展示工具白名单中可公开的查询字段。
        normalized = tool.args_schema.model_validate(arguments).model_dump()
        emit({"kind": "tool", "phase": "started", "arguments": normalized})
        message = tool.invoke(
            {"name": tool.name, "args": normalized, "id": call_id, "type": "tool_call"},
            config={"configurable": {"tool_context": replace(context, emit=emit)}},
        )
        if not isinstance(message, ToolMessage) or not isinstance(message.artifact, EvidenceBundle):
            raise TypeError(f"工具 {tool.name} 未返回约定的证据 artifact")
        bundle = message.artifact
        message.status = "error" if bundle.status == "error" else "success"
        emit({"kind": "tool", "phase": "finished", "status": bundle.status,
              "count": bundle.count, "warnings": bundle.warnings,
              "elapsed": round(time.monotonic() - started, 3)})
        return message
    except Exception as exc:
        # ValidationError 默认字符串会包含原始输入，避免意外记录未声明的敏感值。
        error = "；".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()) if isinstance(exc, ValidationError) else str(exc)[:500]
        emit({"kind": "tool", "phase": "failed", "error_type": type(exc).__name__,
              "error": error, "elapsed": round(time.monotonic() - started, 3)})
        raise
