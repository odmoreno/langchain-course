from dotenv import load_dotenv

# HumanMessage nos permite inicializar mensajes del usuario (rol "human")
from langchain_core.messages import HumanMessage

# MessagesState -> guarda la conversación (lista de mensajes)
# StateGraph -> objeto principal para armar el grafo de nodos
# END -> constante especial que marca el final del grafo
from langgraph.graph import MessagesState, StateGraph, END

# Importamos nuestros nodos definidos en nodes.py
# - run_agent_reasoning: el nodo de razonamiento (LLM)
# - tool_node: el nodo que ejecuta herramientas
from nodes import run_agent_reasoning, tool_node

# Cargamos las variables de entorno (.env), como claves API
load_dotenv()

# Constantes para mayor legibilidad
AGENT_REASON = "agent_reason"  # nombre del nodo de razonamiento
ACT = "act"                    # nombre del nodo de ejecución de tools
LAST = -1                      # en Python -1 = último elemento de una lista


# --- Función de control de flujo ---
def should_continue(state: MessagesState) -> str:
    """
    Decide qué nodo ejecutar después de agent_reason.
    - Si el último mensaje NO tiene tool_calls -> terminamos (END).
    - Si el último mensaje SÍ tiene tool_calls -> vamos al nodo ACT.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


# --- Definición del grafo ---
flow = StateGraph(MessagesState)  # Creamos el grafo que manejará MessagesState

# Agregamos el nodo de razonamiento (llamado agent_reason)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)  # El grafo empieza siempre en agent_reason

# Agregamos el nodo de tools (ejecuta las herramientas)
flow.add_node(ACT, tool_node)

# Conectamos agent_reason con sus posibles salidas:
# - Si la función should_continue devuelve END -> vamos a END
# - Si devuelve ACT -> vamos a ACT
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {END: END, ACT: ACT}
)

# Conectamos el nodo ACT de vuelta a agent_reason
# Esto permite el bucle: ejecutar una tool -> volver a razonar
flow.add_edge(ACT, AGENT_REASON)

# Compilamos el grafo en un "app" listo para ejecutarse
app = flow.compile()

# Generamos un diagrama PNG del grafo (lo que viste en el tutorial)
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# --- Ejecución ---
if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")

    # Invocamos el grafo con un mensaje inicial del usuario
    res = app.invoke({
        "messages": [
            HumanMessage(
                content="What is the temperature in Tokyo? List it and then triple it. Also give me others details like condition, wind speed, humidity, etc")
        ]
    })

    # Imprimimos el contenido del último mensaje (respuesta del agente)
    print(res["messages"][LAST].content)
