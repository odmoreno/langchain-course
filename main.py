
from dotenv import load_dotenv

# Importa 'hub' desde LangChain, que permite acceder a plantillas predefinidas
# alojadas en el LangChain Hub (repositorio de prompts y flujos preconstruidos)
from langchain import hub

# Importa el modelo de lenguaje de Google (Gemini) a través del conector oficial de LangChain
# Este modelo se usará como el motor LLM para generar y razonar sobre texto
from langchain_google_genai import ChatGoogleGenerativeAI

# Importa funciones para crear agentes del tipo ReAct y su ejecutor.
# Un agente ReAct puede "razonar" y "actuar", combinando pensamiento lógico con ejecución de herramientas.
from langchain.agents import create_react_agent, AgentExecutor

# Importa una herramienta experimental de LangChain que permite ejecutar código Python
# dentro del entorno actual (REPL: Read-Eval-Print Loop)
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    """Función principal del script que configura y ejecuta un agente LangChain
    capaz de generar código Python, ejecutarlo y usar los resultados para responder preguntas.
    """
    print("start ...")

    # Instrucciones que definen el comportamiento general del agente.
    # Aquí se le indica que debe escribir y ejecutar código Python cuando sea posible.
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    # Descarga desde LangChain Hub una plantilla base llamada "react-agent-template"
    # Esta plantilla define cómo el agente debe razonar y actuar usando el patrón ReAct (Reason + Act)
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    # Combina las instrucciones personalizadas con la plantilla base
    # De esta manera, el agente tendrá un prompt adaptado con nuestro comportamiento deseado
    prompt = base_prompt.partial(instructions=instructions)

    # Define las herramientas que el agente podrá usar.
    # En este caso, solo una: el Python REPL Tool, que permite ejecutar código Python.
    tools = [PythonREPLTool()]

    # Crea el agente ReAct combinando:
    # - El modelo de lenguaje (Gemini 2.5 Flash)
    # - El prompt con instrucciones
    # - Las herramientas disponibles
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash"),
        tools=tools,
    )

    # Crea un ejecutor (AgentExecutor) que gestiona el ciclo completo del agente:
    # interpretación del prompt, razonamiento, ejecución de herramientas y retorno de resultados
    # 'verbose=True' hace que se muestren en consola los pasos internos del agente (pensamientos, acciones, etc.)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoca el agente con una tarea específica:
    # Generar 15 códigos QR que apunten a la URL de Udemy del curso de LangChain.
    # Cada QR se debe guardar dentro de una carpeta llamada '/qrcode'.
    # Se asume que la librería 'qrcode' ya está instalada en el entorno.
    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain and save it in a folder called /qrcode, you have qrcode package installed already"""
        }
    )


if __name__ == "__main__":
    main()
