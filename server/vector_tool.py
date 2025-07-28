from mcp_decorator import tool

from search_tool import live_web_search  # Import your web search function
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in .env or your shell
GROQ_MODEL = "mistral-saba-24b"

def call_groq_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Init HF embedding model (must match data_loader.py)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Init Qdrant client
client = QdrantClient(host="localhost", port=6333)
@tool
def vector_search(query: str, top_k: int = 3) -> str:
    """Search vector DB for similar documents based on query. Fallback to web search if not found."""

    query_embedding = embed_model.get_text_embedding(query)

    results = client.search(
        collection_name="research",
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )

    if not results:
        print("❌ No match in local knowledge base. Falling back to web search...")
        return live_web_search(query)  # 👈 web search fallback

    return "\n\n".join([point.payload.get("text", "[No Text]") for point in results])
