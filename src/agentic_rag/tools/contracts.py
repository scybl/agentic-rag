"""工具的公开输入约束、私有执行上下文与证据结果。

工具不接收整个 GraphState，也不替调度器重试、写检查点或决定下一步。
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

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
    caller: str = ""
    reason: str = ""
    evidence: Mapping[str, Document] = field(default_factory=dict)
    read_version: Callable[[str], Document] | None = None
    output_schema: type[BaseModel] | None = None


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
    candidates: list[dict] = field(default_factory=list)
    retrieval_complete: bool | None = None

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
                              "candidate_count": len(self.candidates),
                              "deferred_count": sum(not c.get("selected_for_reading", False) for c in self.candidates),
                              "retrieval_complete": self.retrieval_complete,
                              "candidate_preview": [{
                                  "article_id": str(c.get("article_id") or "")[:200],
                                  "title": str(c.get("title") or "")[:200],
                                  "priority": c.get("priority", {}),
                                  "selected_for_reading": c.get("selected_for_reading", False),
                                  "selection_reason": str(c.get("selection_reason") or "")[:200],
                              } for c in self.candidates[:5]],
                              "preview_only": True}, ensure_ascii=False)
        return content, self


@dataclass
class RuleBundle:
    """确定性规则工具的结构化结果，不把未通过伪装成执行异常。"""

    rule: str
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.checks)

    @property
    def warnings(self) -> list[str]:
        return self.violations

    @property
    def status(self) -> str:
        return "ok" if self.passed else "rejected"

    def as_response(self) -> tuple[str, "RuleBundle"]:
        content = json.dumps({
            "rule": self.rule,
            "passed": self.passed,
            "checks": self.checks,
            "violations": self.violations,
            "suggested_queries": self.suggested_queries,
        }, ensure_ascii=False)
        return content, self


@dataclass
class ModelRetryPlan:
    """模型失败后的确定性资源/策略调整，不由失败的模型自行拍脑袋决定。"""

    action: str
    retry: bool
    reason: str
    next_num_predict: int
    next_num_ctx: int
    next_timeout_seconds: int
    next_reasoning: bool
    prompt_feedback: str = ""

    @property
    def count(self) -> int:
        return 1

    @property
    def warnings(self) -> list[str]:
        return []

    @property
    def status(self) -> str:
        if self.action == "split_input":
            return "redirected"
        return "adjusted" if self.retry else "stopped"

    def as_response(self) -> tuple[str, "ModelRetryPlan"]:
        content = json.dumps({
            "action": self.action,
            "retry": self.retry,
            "reason": self.reason,
            "next": {
                "num_predict": self.next_num_predict,
                "num_ctx": self.next_num_ctx,
                "timeout_seconds": self.next_timeout_seconds,
                "reasoning": self.next_reasoning,
            },
            "prompt_feedback": self.prompt_feedback,
        }, ensure_ascii=False)
        return content, self
