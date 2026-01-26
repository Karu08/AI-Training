from langchain_aws import ChatBedrockConverse
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

from tools.rag import hr_policy_rag
from tools.websearch import web_search
from mcp.mcp_client import load_mcp_tools


mcp_tools = load_mcp_tools()


llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an internal research assistant for XYZ Company.

You have access to the following tools:
- HR_Policy_RAG: answers questions from HR policy documents
- Web_Search: fetches industry benchmarks and regulations
- google_docs_search: searches internal Google Docs for INSURANCE related queries

RULES:
- You DO have access to these tools.
- If a question requires documents or external info, you MUST use a tool.
- Never say you lack access to tools.
- Use tools silently and return a final answer.
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


tools = [
    Tool(
        name="HR_Policy_RAG",
        func=hr_policy_rag,
        description="Answer HR policy questions from internal PDF documents"
    ),
    Tool(
        name="Web_Search",
        func=web_search,
        description="xSearch the web for trends and benchmarks"
    ),
    *mcp_tools
]


agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


if __name__ == "__main__":
    while True:
        print("********* AI Agent **********")
        q = input("\nAsk a question (or 'exit'): ")

        if q.lower() == "exit":
            break

        result = agent_executor.invoke({"input": q})
        output = result["output"]

        if isinstance(output, list):
            output = "".join(chunk.get("text", "") for chunk in output)

        print("\nAnswer:\n", output)


