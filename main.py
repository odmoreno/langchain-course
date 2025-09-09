from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)

if __name__ == "__main__":
    main()
