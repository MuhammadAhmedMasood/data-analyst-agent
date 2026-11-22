"""Chat with the agent in the terminal.
Usage: uv run python -m app.cli data/samples/Superstore.csv"""
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
    elif isinstance(message, AIMessage):
        print(f"\nAgent: {message.content}\n")


def main() -> None:
    store = DataStore()
    for path in sys.argv[1:]:
        print(f"Loaded table: {store.load_csv(path)}")
    print(f"Charts will be saved in: {store.chart_dir}")

    graph = build_graph(store, get_llm())
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        history.append(HumanMessage(question))

        for update in graph.stream(
            {"messages": history},
            stream_mode="updates",
            config={"recursion_limit": 20},
        ):
            for node_output in update.values():
                for message in node_output["messages"]:
                    show(message)
                    history.append(message)


if __name__ == "__main__":
    main()