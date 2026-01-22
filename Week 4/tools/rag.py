from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


client = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index_name = PINECONE_INDEX_NAME  

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def hr_policy_rag(query:str)->str:
    """
    Gets relevant content from provided HR Policy document.

    Args:
        user query: string

    Returns:
        a string with the relevant response
    """

    relevant_docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in relevant_docs)