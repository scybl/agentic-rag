"""组装智能体 RAG 流程图。

默认开启研究模式：建立运行 -> 规划 -> 长期记忆召回 -> 并行收集数据源
-> 相关性筛选 -> 持久化分段阅读 -> 证据评估 -> 并行专题分析 -> 生成 -> 核验。
缺口触发有界补搜后回到筛选；核验不通过则补证据或重写。每个节点有持久化检查点。
研究模式关闭时跳过长期记忆、独立阅读与专题调度，保留基础 RAG 流程。
"""

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState
from ..config import settings
from ..research import service
from ..token_usage import meter_node


def build_graph(*, checkpointer=None, research=None):
    research = settings.research_enabled if research is None else research
    workflow = StateGraph(GraphState)

    graph_nodes = {
        "route": nodes.route,
        "collect_sources": nodes.collect_sources,
        "grade_documents": nodes.grade_documents,
        "assess_evidence": nodes.assess_evidence,
        "supplement_sources": nodes.supplement_sources,
        "revise_answer": nodes.revise_answer,
        "generate": nodes.generate,
        "evaluate_generation": nodes.evaluate_generation,
    }

    if research:
        graph_nodes.update({
            "initialize_research": service.initialize,
            "recall_memory": service.recall,
            "read_documents": service.read_documents,
            "dispatch_specialists": service.dispatch_specialists,
        })
    for name, node in graph_nodes.items():
        workflow.add_node(name, meter_node(name, node))

    if research:
        workflow.add_edge(START, "initialize_research")
        workflow.add_edge("initialize_research", "route")
        workflow.add_edge("route", "recall_memory")
        workflow.add_edge("recall_memory", "collect_sources")
        workflow.add_edge("collect_sources", "grade_documents")
        workflow.add_edge("grade_documents", "read_documents")
        workflow.add_edge("read_documents", "assess_evidence")
        workflow.add_conditional_edges("dispatch_specialists", lambda state: state["next_action"],
                                       {"generate": "generate", "supplement": "supplement_sources"})
    else:
        workflow.add_edge(START, "route")
        workflow.add_edge("route", "collect_sources")
        workflow.add_edge("collect_sources", "grade_documents")
        workflow.add_edge("grade_documents", "assess_evidence")
    workflow.add_conditional_edges("assess_evidence", lambda state: state["next_action"],
                                   {"generate": "dispatch_specialists" if research else "generate", "supplement": "supplement_sources"})
    workflow.add_edge("supplement_sources", "grade_documents")
    workflow.add_edge("generate", "evaluate_generation")
    workflow.add_edge("revise_answer", "evaluate_generation")
    workflow.add_conditional_edges(
        "evaluate_generation",
        nodes.decide_after_generation,
        {"finish": END, "supplement": "supplement_sources", "revise": "revise_answer"},
    )

    return workflow.compile(checkpointer=checkpointer)
