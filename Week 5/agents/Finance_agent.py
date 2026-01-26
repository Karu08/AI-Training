from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from web_tools.websearch import web_search


def Finance_Agent(query: str):
    llm = ChatBedrockConverse(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        temperature=0.3
    )

    vectordb = Chroma(
        persist_directory="chroma_Finance",
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    ans = qa_chain.run(query)

    if len(ans.strip()) < 30 or "no information" in ans.lower() or "not mentioned" in ans.lower() or "not provided" in ans.lower():
        web_results = web_search(query)
        return f"{ans}\n\nWeb Results:\n{web_results}"

    return ans



