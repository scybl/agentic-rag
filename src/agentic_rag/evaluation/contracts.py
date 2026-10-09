"""冻结语料、财经问题、相关性标注与实验配置。"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from ..telemetry.schema import Contract


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]


class FrozenDocument(Contract):
    document_id: Nonempty
    title: Nonempty
    summary: Nonempty
    body: Nonempty
    published_at: str | None = None
    source: Nonempty

    @model_validator(mode="after")
    def timestamp(self):
        if self.published_at is not None:
            from ..news_plan import parse_news_timestamp
            if not self.published_at or parse_news_timestamp(self.published_at) is None:
                raise ValueError("日期应为带时区的 ISO 时间或 null")
        return self


class Qrel(Contract):
    document_id: Nonempty
    relevance: Annotated[StrictInt, Field(ge=0, le=3)]


class FinanceCase(Contract):
    case_id: Nonempty
    question: Nonempty
    category: Literal["fact", "time_sensitive", "entity", "multi_hop", "forecast", "probability",
                      "unanswerable", "conflict", "numeric_unit", "unknown_date"]
    split: Literal["dev", "test"]
    answerable: bool
    qrels: list[Qrel]
    reference_answer: Nonempty
    key_evidence: list[Nonempty] = Field(min_length=1)
    forbidden_claims: list[Nonempty] = Field(min_length=1)
    risk_obligations: list[Nonempty]
    rubric: list[Nonempty] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self):
        if len({q.document_id for q in self.qrels}) != len(self.qrels):
            raise ValueError("同一问题的 qrels 不能重复")
        if self.answerable and not any(q.relevance > 0 for q in self.qrels):
            raise ValueError("可回答样本必须提供正相关标注")
        return self


class FinanceSuite(Contract):
    schema_version: Literal[1] = 1
    suite_id: Nonempty
    version: Nonempty
    provenance: Nonempty
    synthetic: bool
    documents: list[FrozenDocument] = Field(min_length=1)
    cases: list[FinanceCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self):
        documents = {d.document_id for d in self.documents}
        if len(documents) != len(self.documents):
            raise ValueError("材料 ID 重复")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("问题 ID 重复")
        if any(q.document_id not in documents for c in self.cases for q in c.qrels):
            raise ValueError("qrels 引用了快照中不存在的材料")
        return self

    @property
    def fingerprint(self):
        return digest(self.model_dump(mode="json"))


class ExperimentConfig(Contract):
    adapter: Literal["current_tfidf", "bm25", "dense", "hybrid", "rerank"] = "current_tfidf"
    top_k: Annotated[StrictInt, Field(ge=1, le=1000)] = 4
    repeat: Annotated[StrictInt, Field(ge=1, le=100)] = 2
    split: Literal["dev", "test", "all"] = "dev"
    sort_by: Literal["relevance", "newest", "oldest"] = "relevance"
    candidate_k: Annotated[StrictInt, Field(ge=1, le=1000)] = 100
    rrf_constant: Annotated[StrictInt, Field(ge=1)] = 60
    rerank_k: Annotated[StrictInt, Field(ge=1, le=1000)] = 80
    embedding_model: str | None = None
    query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    reranker_model: str | None = None
    reranker_timeout: float = Field(default=60, gt=0, le=600)
    mmr: bool = False
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def dependencies(self):
        if (self.adapter in {"dense", "hybrid", "rerank"} or self.mmr) and not self.embedding_model:
            raise ValueError("该实验需要指定本地 embedding_model")
        if self.adapter == "rerank" and not self.reranker_model:
            raise ValueError("重排实验需要指定本地 reranker_model；不存在时会显式降级")
        if self.adapter != "current_tfidf" and self.sort_by != "relevance":
            raise ValueError("新增检索适配器只对照相关性排序")
        if self.adapter == "current_tfidf" and self.mmr:
            raise ValueError("current_tfidf 是冻结基线，不应用 MMR")
        return self


def load_suite(path) -> FinanceSuite:
    return FinanceSuite.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))
