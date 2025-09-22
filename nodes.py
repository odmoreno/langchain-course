from dotenv import load_dotenv
# Cargar variables de entorno (API keys, configuraciones) desde el archivo .env

from langgraph.graph import MessagesState
# MessagesState: objeto que guarda el estado de la conversación
# (lista de mensajes entre usuario, agente y tools)

from langgraph.prebuilt import ToolNode
# ToolNode: nodo preconstruido que ejecuta herramientas si el LLM devuelve un `tool_call`

from react import llm, tools
# Importamos:
# - llm: el modelo de lenguaje configurado en react.py (ya tiene bind_tools)
# - tools: la lista de herramientas disponibles (ej: search, triple, etc.)

load_dotenv()
# Activamos la carga de variables de entorno (.env)

# --- Mensaje de sistema ---
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""
# Este mensaje define el rol del asistente.
# El LLM siempre lo recibe antes de procesar las instrucciones del usuario.
# Sirve para recordarle: "eres un asistente útil con acceso a herramientas".

# --- Nodo de razonamiento ---
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Nodo que maneja el razonamiento del agente.
    El LLM decide si contesta directamente o si necesita llamar a una tool.
    """
    response = llm.invoke(
        [  # Lista de mensajes que recibe el LLM:
            {"role": "system", "content": SYSTEM_MESSAGE},  # contexto del sistema
            *state["messages"]  # historial de la conversación hasta ahora
        ]
    )
    # El LLM devuelve un mensaje (AIMessage o tool_call)
    return {"messages": [response]}
    # Se retorna en forma de diccionario, para que LangGraph actualice el estado.

# --- Nodo de herramientas ---
tool_node = ToolNode(tools)
# Este nodo ejecuta las herramientas definidas en `react.py`.
# Si el último mensaje del LLM incluye un tool_call,
# ToolNode corre la función correspondiente y devuelve el resultado en los mensajes.
