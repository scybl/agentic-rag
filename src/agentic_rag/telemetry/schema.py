"""内部 Trace 协议，不绑定某个厂商的语义属性版本。"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def identity(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:32]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Span(Contract):
    span_id: str = Field(min_length=1)
    parent_id: str | None = None
    name: str
    kind: Literal["run", "session", "node", "task", "llm", "tool", "retriever"]
    status: Literal["pending", "completed", "failed", "interrupted", "unknown"] = "unknown"
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    elapsed_seconds: float | None = Field(default=None, ge=0)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class Event(Contract):
    event_id: str
    span_id: str
    name: str
    at: float | None = Field(default=None, ge=0)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class Trace(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    spans: list[Span]
    events: list[Event] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_tree(self):
        spans = {s.span_id: s for s in self.spans}
        if len(spans) != len(self.spans):
            raise ValueError("Span ID 重复")
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("Event ID 重复")
        if sum(s.parent_id is None for s in self.spans) != 1:
            raise ValueError("Trace 必须有且仅有一个根")
        if any((s.parent_id is None) != (s.kind == "run") for s in self.spans):
            raise ValueError("唯一根必须为 run，其他节点不能是 run")
        resolved = set()
        for span in self.spans:
            path, node = set(), span
            while node.span_id not in resolved:
                if node.span_id in path:
                    raise ValueError("Span 父子关系形成循环")
                path.add(node.span_id)
                if node.parent_id is None:
                    break
                if node.parent_id not in spans:
                    raise ValueError("Span 引用不存在的父节点")
                node = spans[node.parent_id]
            resolved.update(path)
        if any(e.span_id not in spans for e in self.events):
            raise ValueError("Event 引用不存在的 Span")
        return self
