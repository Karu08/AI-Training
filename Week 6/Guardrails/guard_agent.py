from nemoguardrails import LLMRails, RailsConfig
import os
from rag_agent import run_rag


config_path = os.path.join(os.path.dirname(__file__))
config = RailsConfig.from_path(config_path)
rails = LLMRails(config)

def guarded_rag_query(user_input: str):
    """
    Run Guardrails first, then call RAG chain if safe.
    """
    
    guard_response = rails.generate(messages=[{
        "role": "user",
        "content": user_input
    }])

    
    if isinstance(guard_response, dict):
        guard_text = guard_response.get("content", "").strip()
    else:
        guard_text = str(guard_response).strip()

    blocked_phrases = [
        "I can’t help with that request.",
        "I can’t follow that instruction.",
        "I can answer only IT-related questions."
    ]

    if any(phrase in guard_text for phrase in blocked_phrases):
        return guard_text 
    
    
    rag_answer = run_rag(user_input)
    if not rag_answer.strip():
        rag_answer = "I can answer only IT-related questions."

    return rag_answer


print("IT Support Assistant (Guardrails Enabled)")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Assistant: Bye!")
        break

    response = guarded_rag_query(user_input)
    print(f"Assistant: {response}\n")
