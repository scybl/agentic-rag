"""工具的公开输入约束、私有执行上下文与证据结果。

工具不接收整个 GraphState，也不替调度器重试、写检查点或决定下一步。
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, StringConstraints


Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ToolInput(BaseModel):
    """拒绝未声明参数，避免把数据库路径或认证信息当作模型输入。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


@dataclass(frozen=True)
class ToolContext:
    """程序注入的能力，不进入工具给模型看的参数 Schema。

    原文工具只能读取本次给定的证据版本，模型不能指定任意文件或数据库。
    """

    emit: Callable[[dict], None] = lambda _event: None
    evidence: Mapping[str, Document] = field(default_factory=dict)
    read_version: Callable[[str], Document] | None = None


def context_from(config: RunnableConfig) -> ToolContext:
    context = config.get("configurable", {}).get("tool_context")
    return context if isinstance(context, ToolContext) else ToolContext()


@dataclass
class EvidenceBundle:
    """完整证据通过 artifact 给程序；content 只提供有界的观察摘要。"""

    documents: list[Document] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.documents) + len(self.passages)

    @property
    def status(self) -> str:
        if self.warnings:
            return "degraded" if self.count else "error"
        return "ok" if self.count else "empty"

    def as_response(self) -> tuple[str, "EvidenceBundle"]:
        # 完整正文和新闻事件不重复塞入工具消息；保留可辨认的来源与摘要。
        preview = [{"source": str(d.metadata.get("source", ""))[:400],
                    "title": str(d.metadata.get("title", ""))[:200],
                    "content_kind": str(d.metadata.get("content_kind", ""))[:60],
                    "excerpt": d.page_content[:300]} for d in self.documents[:5]]
        passages = [{**p, "quote": p["quote"][:200],
                     "end": p["start"] + min(200, len(p["quote"])),
                     "excerpted": len(p["quote"]) > 200} for p in self.passages[:8]]
        content = json.dumps({"status": self.status, "count": self.count,
                              "warnings": [str(w)[:500] for w in self.warnings[:5]],
                              "documents": preview, "passages": passages,
                              "preview_only": True}, ensure_ascii=False)
        return content, self
