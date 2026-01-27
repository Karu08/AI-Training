from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun


def search_Finance_docs(query: str)->str:
    """
    Searches in internal documnents for Finance-related queries.

    Args:
        user query: string
    
    Returns:
        The result for user query as a string

    """
    
    llm = ChatBedrockConverse(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        temperature=0.3
    )

    system_prompt = """
    You are a Finance helpdesk assistant.

    RULES:
    - Give direct steps or factual answers only.
    - Do NOT summarize, comment, or explain policies.
    - Do NOT mention documents, sources, or context.
    - Return only the final answer in plain text.
    - If unsure, just say you don’t know.
    """

    vectordb = Chroma(
        persist_directory="chroma_Finance",
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
    chain_type_kwargs={"prompt": system_prompt})

    ans = qa_chain.run(query)
    return ans


def web_search(query:str)->str:
    """
    Does a web search to return results for general user queries

    Args:
        query: string
    
    Returns:
        the result for user query as a string
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

