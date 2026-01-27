from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrockConverse
from Tools.Finance_tool import search_Finance_docs
from Tools.Finance_tool import web_search
from langchain.tools import Tool


llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an Finance helpdesk assistant.

Answer the user's question with clear, step-by-step instructions.

RULES:
- Give direct steps only.
- Do NOT explain policies.
- Do NOT mention documents, policies, or sources.
- Do NOT add qualifiers or commentary.
- Return **only the final answer** in plain text.
"""),
    ("human", "{input}"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


tools = [
    Tool(
        name="Finance_Policy_RAG",
        func=search_Finance_docs,
        description="Answer HR policy questions from internal documents"
    ),
    Tool(
        name="Web_Search",
        func=web_search,
        description="Search the web for trends and benchmarks"
    )
]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

finance_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=False
)

def Finance_Agent(query: str) -> str:
    result = finance_agent_executor.invoke({"input": query})
    return result["output"]
  

