from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]
# llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.1)
# gemma3:270m
llm = ChatOllama(temperature=0, model="llama3.1:8b").bind(
    stop=["\nObservation:"])
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=4)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        },
    )
    print(result)

if __name__ == "__main__":
    main()
