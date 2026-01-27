from langchain_aws import ChatBedrockConverse
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.types import TraceContext
import time
import os

load_dotenv()
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
)

llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    temperature=0.3
)


def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Use this for math calculations"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


print("Calculator Agent started. Type 'exit' to quit.\n")

while True:
    user_input = input("User query: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    trace_context = TraceContext()
    with langfuse.start_as_current_span(
        trace_context=trace_context,
        name="agent-interaction",
        input={"user_query": user_input}
    ) as root_span:

        # You can optionally propagate user/session attributes too
        root_span.update_trace(
            user_id="user_local",
            session_id="session_local"
        )

        # Log start time
        start = time.time()

        # Tool call span
        with langfuse.start_as_current_span(name="calculator-tool", input={"expression": user_input}):
            tool_output = None
            try:
                tool_output = calculator(user_input)
            except Exception as tool_err:
                tool_output = str(tool_err)

        # Agent run
        try:
            response = agent.run(user_input)
        except Exception as e:
            response = f"Agent error: {e}"

        latency = time.time() - start

  
        root_span.update(
            output={"response": response},
            metadata={"latency_sec": latency}
        )

        print(f"Agent: {response}\n")

 