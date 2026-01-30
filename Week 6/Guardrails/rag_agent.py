from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA


docs = TextLoader("/Users/karunyarv/AI-Training/Guardrails/IT_policy.txt").load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

def run_rag(user_query: str):
    try:
        response = qa_chain.run(user_query)
        return response.strip()
    except Exception:
        return "I can answer only IT-related questions."
