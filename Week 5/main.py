from graph import support_graph

while True:
    query = input("\nAsk your question (or 'exit'): ")
    if query.lower() == "exit":
        break

    result = support_graph.invoke({"query": query})
    print("\nAnswer:\n", result["answer"])
