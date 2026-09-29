"""组装智能体 RAG 流程图。

默认开启研究模式：建立运行 -> 规划 -> 长期记忆召回 -> 并行收集数据源
-> 持久化分段阅读 -> 相关性筛选 -> 证据评估 -> 并行专题分析 -> 生成 -> 核验。
缺口触发有界补搜后回到阅读；核验不通过则补证据或重写。每个节点有持久化检查点。
研究模式关闭时跳过长期记忆、独立阅读与专题调度，保留基础 RAG 流程。
"""

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState
from ..config import settings
from ..research import service


def build_graph(*, checkpointer=None, research=None):
    research = settings.research_enabled if research is None else research
    workflow = StateGraph(GraphState)

    workflow.add_node("route", nodes.route)
    workflow.add_node("collect_sources", nodes.collect_sources)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("assess_evidence", nodes.assess_evidence)
    workflow.add_node("supplement_sources", nodes.supplement_sources)
    workflow.add_node("revise_answer", nodes.revise_answer)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("evaluate_generation", nodes.evaluate_generation)

    if research:
        workflow.add_node("initialize_research", service.initialize)
        workflow.add_node("recall_memory", service.recall)
        workflow.add_node("read_documents", service.read_documents)
        workflow.add_node("dispatch_specialists", service.dispatch_specialists)
        workflow.add_edge(START, "initialize_research")
        workflow.add_edge("initialize_research", "route")
        workflow.add_edge("route", "recall_memory")
        workflow.add_edge("recall_memory", "collect_sources")
        workflow.add_edge("collect_sources", "read_documents")
        workflow.add_edge("read_documents", "grade_documents")
        workflow.add_conditional_edges("dispatch_specialists", lambda state: state["next_action"],
                                       {"generate": "generate", "supplement": "supplement_sources"})
    else:
        workflow.add_edge(START, "route")
        workflow.add_edge("route", "collect_sources")
        workflow.add_edge("collect_sources", "grade_documents")
    workflow.add_edge("grade_documents", "assess_evidence")
    workflow.add_conditional_edges("assess_evidence", lambda state: state["next_action"],
                                   {"generate": "dispatch_specialists" if research else "generate", "supplement": "supplement_sources"})
    workflow.add_edge("supplement_sources", "read_documents" if research else "grade_documents")
    workflow.add_edge("generate", "evaluate_generation")
    workflow.add_edge("revise_answer", "evaluate_generation")
    workflow.add_conditional_edges(
        "evaluate_generation",
        nodes.decide_after_generation,
        {"finish": END, "supplement": "supplement_sources", "revise": "revise_answer"},
    )

    return workflow.compile(checkpointer=checkpointer)
