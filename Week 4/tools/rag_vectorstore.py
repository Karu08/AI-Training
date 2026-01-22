from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
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


loader = PyPDFLoader("/Users/karunyarv/Documents/AITraining/Week 4/docs/HRPolicy.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)


vectorstore = PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name=index_name
)
print("Rag PDF indexed successfully")
