from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os

from tools.rag import hr_policy_rag
from tools.websearch import web_search
from mcp.mcp_client import load_mcp_tools

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

mcp_tools = load_mcp_tools()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

tools = [
    Tool(
        name="HR_Policy_RAG",
        func=hr_policy_rag,
        description="Answer HR policy questions from internal PDF documents"
    ),
    Tool(
        name="Web_Search",
        func=web_search,
        description="Search the web for trends and benchmarks"
    ),
    *mcp_tools
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

if __name__ == "__main__":
    while True:
        print("********* AI Agent **********")
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", agent.run(q))
