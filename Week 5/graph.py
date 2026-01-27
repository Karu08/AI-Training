from langgraph.graph import StateGraph, END
from agents.Supervisor import supervisor_agent
from agents.IT_agent import IT_Agent
from agents.Finance_agent import Finance_Agent

class SupportState(dict):
    pass

def supervisor_node(state):
    category = supervisor_agent(state["query"])
    state["category"] = category
    print("\nState: ",state)
    return state

def it_node(state):
    state["answer"] = IT_Agent(state["query"])
    return state

def finance_node(state):
    state["answer"] = Finance_Agent(state["query"])
    return state

graph = StateGraph(SupportState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("it_agent", it_node)
graph.add_node("finance_agent", finance_node)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    lambda state: state["category"],
    {
        "IT": "it_agent",
        "Finance": "finance_agent"
    }
)

graph.add_edge("it_agent", END)
graph.add_edge("finance_agent", END)

support_graph = graph.compile()
