from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from grounded_video_agent.agent.state import AgentState

from .dependencies import AgentDependencies
from .nodes import AgentNodes
from .routing import route_from_state


def build_agent_graph(
    dependencies: AgentDependencies,
    *,
    checkpointer: Any = None,
) -> Any:
    nodes = AgentNodes(dependencies)
    graph = StateGraph(AgentState)
    graph.add_node("bootstrap", nodes.bootstrap)
    graph.add_node("plan", nodes.plan)
    graph.add_node("guard", nodes.guard)
    graph.add_node("execute_tool", nodes.execute_tool)
    graph.add_node("answer_gate", nodes.answer_gate)
    graph.add_node("draft_answer", nodes.draft_answer)
    graph.add_node("verify", nodes.verify)
    graph.add_node("deliver", nodes.deliver)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        route_from_state,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "plan",
        route_from_state,
        {"guard": "guard", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "guard",
        route_from_state,
        {
            "plan": "plan",
            "execute_tool": "execute_tool",
            "answer_gate": "answer_gate",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_tool", "plan")
    graph.add_conditional_edges(
        "answer_gate",
        route_from_state,
        {
            "plan": "plan",
            "draft_answer": "draft_answer",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "draft_answer",
        route_from_state,
        {"verify": "verify", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "verify",
        route_from_state,
        {"plan": "plan", "deliver": "deliver", "finalize": "finalize"},
    )
    graph.add_edge("deliver", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name="grounded_video_agent")
