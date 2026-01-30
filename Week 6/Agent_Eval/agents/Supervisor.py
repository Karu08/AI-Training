from langchain_aws import ChatBedrockConverse
from agentevals.trajectory.match import create_trajectory_match_evaluator

trajectory_eval = create_trajectory_match_evaluator()

eval_set = [
    {
        "query": "What softwares can be installed on the company device?",
        "expected_trajectory": [
            {"action": "classification", "output": "Finance"},
            {"action": "routing", "output": "IT_Agent"}
        ]
    },
    {
        "query": "What is the company's payroll processing?",
        "expected_trajectory": [
            {"action": "classification", "output": "Finance"},
            {"action": "routing", "output": "Finance_Agent"}
        ]
    }
]


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



def supervisor_agent_trajectory(query: str):
    label = supervisor_agent(query)

    if label == "IT":
        route = "IT_Agent"
    elif label == "Finance":
        route = "Finance_Agent"
    else:
        route = "Unknown"

    return [
        {"action": "classification", "output": label},
        {"action": "routing", "output": route}
    ]


def evaluate_single_query(user_query, eval_set):
    expected_tr = None
    for case in eval_set:
        if case["query"] == user_query:
            expected_tr = case["expected_trajectory"]
            break

    if expected_tr is None:
        print("No expected trajectory defined for this query")
        return


    actual_tr = supervisor_agent_trajectory(user_query)

    def traj_to_messages(traj):
        messages = []
        for step in traj:
            messages.append({
                "role": "user", 
                "content": f"Action: {step['action']}, Output: {step['output']}"
            })
        return messages

    actual_msgs = traj_to_messages(actual_tr)
    expected_msgs = traj_to_messages(expected_tr)

    
    result = trajectory_eval(
        outputs=actual_msgs,
        reference_outputs=expected_msgs
    )

    print("Query:", user_query)
    print("Actual Trajectory:", actual_msgs)
    print("Expected Trajectory:", expected_msgs)
    print("Evaluation Result:", result)



user_query = "What softwares can be installed on the company device?"
evaluate_single_query(user_query, eval_set)
