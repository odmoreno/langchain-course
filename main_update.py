from typing import List, Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# builder = MessageGraph()
builder = StateGraph(state_schema=MessageGraph)
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


# def event_loop(state: List[BaseMessage]) -> str:
def event_loop(state: MessageGraph) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges(
    "revise", event_loop, {END: END, "execute_tools": "execute_tools"})
builder.set_entry_point("draft")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())


graph.get_graph().draw_mermaid_png(
    output_file_path="graph.png",
    max_retries=5,
    retry_delay=2.0
)


def main():
    print("Hello from langchain-course!")

    query = "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
    res = graph.invoke({"messages": [HumanMessage(content=query)]})

    last_msg = res["messages"][-1]
    # Si fue una llamada a herramienta, quizá haya tool_calls; si no, imprime el contenido
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        # Ojo: según tu chain, puede no existir "answer" aquí; ajusta a tu esquema real
        print(last_msg.tool_calls[0].get(
            "args", {}).get("answer", last_msg.content))
    else:
        print(last_msg.content)
    # print(res[-1].tool_calls[0]["args"]["answer"])
    # print(res)


if __name__ == "__main__":
    main()
