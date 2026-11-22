"""Phase 3 agent: an LLM that loops between thinking and calling tools."""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.data.store import DataStore
from app.tools.chart_tools import make_chart_tools
from app.tools.data_tools import make_data_tools

SYSTEM_PROMPT = """You are a careful data analyst. You answer questions about the
user's tables by calling tools, never by guessing.

How to work:
1. Call get_schema first so you know the exact table and column names.
2. Prefer run_sql for totals, counts, filters, rankings and group-bys.
3. Use run_python for cleaning messy values, statistics or reshaping.
4. Use make_chart when a trend or comparison is clearer as a picture.
5. If a tool returns an error, read it, fix your query or code, and try again.

When you answer:
- Only state numbers that came from a tool result. Round sensibly.
- Say briefly how you got the answer (e.g. "summed Sales by Region").
- Mention any assumptions or data problems you noticed.
- If you made a chart, mention its file name."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(store: DataStore, llm: BaseChatModel):
    tools = make_data_tools(store) + make_chart_tools(store)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        results = []
        for call in last_message.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                output = f"Error: no tool named '{call['name']}'."
            else:
                output = tool.invoke(call["args"])
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results}

    def route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()