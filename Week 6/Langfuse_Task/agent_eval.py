from langchain.agents import initialize_agent, Tool, AgentType
from langchain_aws import ChatBedrockConverse
from eval_data import evaluation_samples
import time


def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [Tool(name="Calculator", func=calculator, description="Use this for math calculations")]

llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    temperature=0.3
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)


results = []

for sample in evaluation_samples:
    query = sample["input"]
    expected = sample["expected_output"]

    start = time.time()
    try:
        response = agent.run(query)
    except Exception as e:
        response = str(e)
    latency = time.time() - start

    
    correct = expected.strip() in response.strip()

    
    tool_used = "Calculator" in response

    results.append({
        "input": query,
        "expected": expected,
        "response": response,
        "correct": correct,
        "latency_sec": round(latency, 3),
        "tool_used": tool_used
    })


# Saving results to markdown
md_lines = ["# Agent Evaluation Report\n"]
for r in results:
    md_lines.append(f"## Input: {r['input']}")
    md_lines.append(f"- Expected: {r['expected']}")
    md_lines.append(f"- Response: {r['response']}")
    md_lines.append(f"- Correct: {r['correct']}")
    md_lines.append(f"- Latency (s): {r['latency_sec']}")
    md_lines.append(f"- Tool Used: {r['tool_used']}\n")

with open("Agent_Evaluation_Report.md", "w") as f:
    f.write("\n".join(md_lines))

print("Agent evaluation complete! Results saved to Agent_Evaluation_Report.md")
