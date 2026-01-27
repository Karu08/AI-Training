from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun


def search_IT_docs(query: str)->str:
    """
    Searches in internal documnents for IT-related queries.

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

    vectordb = Chroma(
        persist_directory="chroma_IT",
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
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

