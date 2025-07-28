from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# ✅ Set Hugging Face embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load documents from local folder
documents = SimpleDirectoryReader("data/docs").load_data()

# ✅ Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# ✅ Create collection explicitly
client.recreate_collection(
    collection_name="research",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 for MiniLM
)

# ✅ Vector store and indexing
vector_store = QdrantVectorStore(client=client, collection_name="research")
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

print("✅ Documents indexed into Qdrant using Hugging Face embeddings!")
