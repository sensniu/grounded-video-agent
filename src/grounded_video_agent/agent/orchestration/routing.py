from __future__ import annotations

from grounded_video_agent.agent.state import AgentState


async def route_from_state(state: AgentState) -> str:
    return state["route"]
