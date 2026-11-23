"""Phase 4 agent: plan -> execute with tools -> check -> retry if needed."""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.data.store import DataStore
from app.tools.chart_tools import make_chart_tools
from app.tools.data_tools import make_data_tools

MAX_ATTEMPTS = 2

PLANNER_PROMPT = """You plan data analyses. You never run anything yourself.
Given the tables, the conversation so far and the user's question, write:
- data_checks: data-quality checks worth running before answering, only for
  columns this question uses. Examples: duplicate IDs, missing values,
  inconsistent spelling/case/whitespace, numbers stored as text, negative or
  impossible values, totals that don't match quantity x price.
- steps: short ordered steps to answer the question, naming the tool for each
  (run_sql, run_python or make_chart).
Keep it short: for a simple question on clean data, 1-3 steps and few checks.
If reviewer feedback is given, write a new plan that fixes every point."""

EXECUTOR_PROMPT = """You are a careful data analyst answering the user's latest
question with tools. Available tables: {tables}

Follow this plan, running the data checks first:

{plan}

Rules:
- Use run_sql for aggregations, run_python for cleaning or statistics, and
  make_chart only when the plan asks for a chart.
- If a tool returns an error, fix your query or code and try again.
- If the reviewer sent feedback, fix every point it raises.

In your final answer:
- Only state numbers that came from tool results.
- Say how you got the answer and which cleaning decisions you made.
- List data problems you found and how you handled them.
- Mention chart file names if you made charts.
- If the question names something that isn't in the data (e.g. a column),
  say what you used instead in your first sentence, or ask the user."""

CHECKER_PROMPT = """You review a data analyst's draft answer before the user
sees it. Judge only from the evidence given; assume nothing else.

Put a problem in blocking_problems only if it is one of these:
- a number in the draft does not appear in, or follow directly from, a tool result
- a data problem that changes the result (duplicates, missing values,
  impossible values such as negative sales) was not checked, or was ignored
  without being mentioned
- a placeholder such as 'N/A', 'unknown', NULL or blank is shown as a real
  category without saying it means missing data
- rows were excluded from the result without saying so
- the draft says the data was cleaned differently from what the SQL or code
  did, in a way that changes what the numbers mean
- the draft does not answer the question

Everything else (wording, level of detail, how explanations are phrased) goes
in minor_notes. A correct, honest answer with imperfect wording must have no
blocking problems. Make each blocking problem specific: what is wrong and
what to do instead."""


class Plan(BaseModel):
    data_checks: list[str] = Field(description="Data-quality checks to run first")
    steps: list[str] = Field(description="Ordered analysis steps, each naming a tool")


class Review(BaseModel):
    blocking_problems: list[str] = Field(
        description="Problems that make a number wrong or unsupported, or would "
        "mislead the user about the result. Empty if there are none."
    )
    minor_notes: list[str] = Field(
        description="Wording or explanation improvements that do not change the result."
    )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    attempts: int
    passed: bool


def is_feedback(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) and message.name == "reviewer"


def since_question(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Everything after the user's latest question (all attempts at answering it)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not is_feedback(messages[i]):
            return messages[i:]
    return messages


def recent_conversation(messages: list[AnyMessage], limit: int = 6) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage) and not is_feedback(m):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and not m.tool_calls and m.content:
            lines.append(f"Analyst: {m.content}")
    return "\n".join(lines[-limit:])


def format_evidence(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                lines.append(f"CALL {call['name']}: {call['args']}")
        elif isinstance(m, ToolMessage):
            lines.append(f"RESULT:\n{m.content[:2000]}")
    return "\n".join(lines)


def build_graph(store: DataStore, llm: BaseChatModel, checkpointer=None):
    tools = make_data_tools(store) + make_chart_tools(store)
    tools_by_name = {t.name: t for t in tools}
    executor_llm = llm.bind_tools(tools, parallel_tool_calls=False)
    planner_llm = llm.with_structured_output(Plan)
    checker_llm = llm.with_structured_output(Review)

    def planner_node(state: AgentState) -> dict:
        messages = state["messages"]
        current = since_question(messages)
        schema = "\n\n".join(store.describe(t) for t in store.tables)
        feedback = messages[-1].content if is_feedback(messages[-1]) else "None"
        prompt = (
            f"Tables:\n{schema}\n\n"
            f"Conversation so far:\n{recent_conversation(messages)}\n\n"
            f"Question: {current[0].content}\n\n"
            f"Reviewer feedback: {feedback}"
        )
        plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(prompt)])
        plan_text = "Data checks:\n" + "\n".join(f"- {c}" for c in plan.data_checks)
        plan_text += "\nSteps:\n" + "\n".join(
            f"{n}. {s}" for n, s in enumerate(plan.steps, start=1)
        )
        return {"plan": plan_text, "attempts": state.get("attempts", 0) + 1}

    def executor_node(state: AgentState) -> dict:
        system = SystemMessage(EXECUTOR_PROMPT.format(plan=state["plan"], tables=", ".join(store.tables)))
        response = executor_llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    def tools_node(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                output = f"Error: no tool named '{call['name']}'."
            else:
                output = tool.invoke(call["args"])
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results}

    def checker_node(state: AgentState) -> dict:
        current = since_question(state["messages"])
        prompt = (
            f"Question: {current[0].content}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Tool calls and results:\n{format_evidence(current)}\n\n"
            f"Draft answer:\n{current[-1].content}"
        )
        review = checker_llm.invoke([SystemMessage(CHECKER_PROMPT), HumanMessage(prompt)])
        if not review.blocking_problems:
            return {"passed": True}

        problems = "\n".join(f"- {p}" for p in review.blocking_problems)
        if state["attempts"] >= MAX_ATTEMPTS:
            note = AIMessage(f"Note: a final review still found problems:\n{problems}")
            return {"passed": False, "messages": [note]}
        feedback = HumanMessage(
            f"Reviewer feedback. Fix these and answer again:\n{problems}",
            name="reviewer",
        )
        return {"passed": False, "messages": [feedback]}

    def route_after_executor(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "checker"

    def route_after_checker(state: AgentState) -> str:
        if state["passed"] or state["attempts"] >= MAX_ATTEMPTS:
            return END
        return "planner"

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("tools", tools_node)
    builder.add_node("checker", checker_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", route_after_executor, ["tools", "checker"])
    builder.add_edge("tools", "executor")
    builder.add_conditional_edges("checker", route_after_checker, ["planner", END])
    return builder.compile(checkpointer=checkpointer)