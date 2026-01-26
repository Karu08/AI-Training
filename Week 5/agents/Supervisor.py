from langchain_aws import ChatBedrockConverse

def supervisor_agent(query: str):
    llm = ChatBedrockConverse(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        temperature=0.3
    )

    system_prompt = f"""
    You are a supervisor agent.

    Classify the user input query as either:
    IT
    Finance

    Respond with ONLY ONE word.

    Query: {query}
    """

    response = llm.invoke(system_prompt)
    return response.content.strip()
