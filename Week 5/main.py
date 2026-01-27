from graph import support_graph

print("***************** AI Agent *****************")
while True:
    query = input("\nAsk your question (or 'exit'): ")
    if query.lower() == "exit":
        break

    result = support_graph.invoke({"query": query})
    answer = result["answer"]
    
    if isinstance(answer, list):
        answer = answer[0]["text"]

    print("\nAnswer:", answer)
