from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
# llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.1)
# gemma3:270m
llm = ChatOllama(temperature=0, model="llama3.1:8b")
# react_prompt = hub.pull("hwchase17/react")

outputparser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=outputparser.get_format_instructions())


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=12)

extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: outputparser.parse(x))
chain = agent_executor | extract_output | parse_output

def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        },
    )
    print(result)

if __name__ == "__main__":
    main()
