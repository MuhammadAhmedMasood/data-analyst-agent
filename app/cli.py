"""Chat with the agent in the terminal.
Usage: uv run python -m app.cli data/samples/Superstore.csv"""
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.agent.llm import get_llm
from app.data.store import DataStore


def show(message) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        for call in message.tool_calls:
            print(f"  -> {call['name']}({call['args']})")
    elif isinstance(message, ToolMessage):
        preview = message.content[:300].replace("\n", "\n     ")
        print(f"  <- {preview}")
    elif isinstance(message, HumanMessage):
        print(f"\n  [reviewer] {message.content}\n")
    elif isinstance(message, AIMessage):
        print(f"\nAgent: {message.content}\n")


def main() -> None:
    store = DataStore()
    for path in sys.argv[1:]:
        print(f"Loaded table: {store.load_csv(path)}")
    print(f"Charts will be saved in: {store.chart_dir}")

    graph = build_graph(store, get_llm(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 40}

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break

        new_input = {"messages": [HumanMessage(question)], "attempts": 0, "passed": False}
        for update in graph.stream(new_input, config=config, stream_mode="updates"):
            for node_name, output in update.items():
                if node_name == "planner":
                    print(f"\n  [plan, attempt {output['attempts']}]\n{output['plan']}\n")
                elif node_name == "checker":
                    print("  [check] passed" if output["passed"] else "  [check] failed")
                for message in output.get("messages", []):
                    show(message)


if __name__ == "__main__":
    main()