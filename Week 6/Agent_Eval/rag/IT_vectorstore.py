from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = TextLoader("/Users/karunyarv/AI-Training/Week 5/data/IT_docs/IT_policy.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectordb = Chroma.from_documents(
    persist_directory="chroma_IT",
    documents=docs,
    embedding=embeddings
)
vectordb.persist()

print("IT vectorstore built successfully.")
