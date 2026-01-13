import boto3
import json
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader


bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1" 
)


def load_documents():
    reader = PdfReader("bankQA.pdf")

    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    full_text = "\n\n".join(pages).lower()
    return full_text

    
# this is char based chunking
# def chunk_text(text, chunk_size=800, overlap=150):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap

#     return chunks

# this is paragraph based chunking
def chunk_text(text):
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks



embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)



def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_chunks(query, index, chunks, k=5):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def build_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)

    return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{user_query}

Answer:
"""

def call_claude(prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.2
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(body),
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def rag_chatbot(question, index, chunks):
    retrieved_chunks = retrieve_chunks(question, index, chunks)
    prompt = build_prompt(retrieved_chunks, question)
    return call_claude(prompt)


if __name__ == "__main__":
    print("Initializing RAG chatbot...")

    text = load_documents()
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)

    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question: ")

        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = rag_chatbot(user_query, index, chunks)
        print("\nAnswer:")
        print(answer)
        print("-" * 60)

