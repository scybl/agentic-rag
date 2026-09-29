"""只回读当前证据中已保存的新闻版本，不允许模型任意访问存储。"""

from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool
from pydantic import Field, StringConstraints

from ..evidence import numbered_sentences
from ..research.store import search_tokens
from .contracts import EvidenceBundle, Query, ToolInput, context_from


EvidenceId = Annotated[str, StringConstraints(pattern=r"^E[1-9]\d*$")]


class NewsReadInput(ToolInput):
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=2, description="当前证据中要回读的E编号，最多两篇；不可传URL、路径或任意版本号")
    goal: Query = Field(description="这次需要核对的具体事实或口径，用于挑选相关原文片段")


@tool("read_news", args_schema=NewsReadInput, response_format="content_and_artifact")
def read_news(evidence_ids: list[str], goal: str, config: RunnableConfig) -> tuple[str, EvidenceBundle]:
    """回读当前证据中已保存新闻的相关原文，返回版本、绝对位置和逐字引用。

    最多两篇、总计最多1600字符；不是再次联网抓取。需程序注入证据白名单。
    摘要来源仍标为摘要；资料未保存时明确说明，不伪装成全文。
    """
    context = context_from(config)
    if context.read_version is None:
        raise ToolException("原文回读需要本次研究的证据上下文")
    ids = list(dict.fromkeys(evidence_ids))
    # 先检查全部请求，防止前一篇已读取后才发现后一篇越界。
    if any(eid not in context.evidence for eid in ids):
        raise ToolException("原文回读请求引用了不存在的证据编号")
    passages, warnings = [], []
    terms = search_tokens(goal)
    for eid in ids:
        document = context.evidence[eid]
        version = document.metadata.get("memory_version")
        if not version:
            warnings.append(f"{eid} 没有已保存的新闻原文版本，只能使用现有资料")
            continue
        original = context.read_version(version).page_content
        sentences, _ = numbered_sentences(original)
        ranked = sorted(sentences.values(), key=lambda p: sum(t in original[p[0]:p[1]].lower() for t in terms), reverse=True)
        remaining = 1600 // len(ids)
        for start, end in ranked:
            if remaining < 50:
                break
            end = min(end, start + remaining)
            quote = original[start:end]
            passages.append({"evidence_id": eid, "version_id": version, "start": start, "end": end,
                             "quote": quote, "content_kind": document.metadata.get("content_kind", "unknown")})
            remaining -= len(quote)
    return EvidenceBundle(passages=passages, warnings=warnings).as_response()
