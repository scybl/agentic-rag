"""稳定方法知识检索；索引生命周期仍由 ingestion 模块维护。"""

from langchain_core.tools import tool
from pydantic import Field

from .contracts import EvidenceBundle, Query, ToolInput


class KnowledgeSearchInput(ToolInput):
    query: Query = Field(description="要寻找的分析方法、领域概念或稳定知识，不用于查询实时新闻")


@tool("search_knowledge", args_schema=KnowledgeSearchInput, response_format="content_and_artifact")
def search_knowledge(query: str) -> tuple[str, EvidenceBundle]:
    """检索本地 knowledge 中的稳定知识和分析方法，返回原始文档片段及来源。

    不负责提供最新新闻；只在实际调用时初始化索引和嵌入，条数使用 RETRIEVAL_K。
    """
    from ..ingestion import ensure_index, get_retriever

    ensure_index()
    documents = get_retriever().invoke(query)
    documents = [doc.model_copy(update={"metadata": {**doc.metadata, "source_type": "vectorstore"}})
                 for doc in documents]
    return EvidenceBundle(documents=documents).as_response()
