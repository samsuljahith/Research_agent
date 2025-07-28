from fastapi import FastAPI
from vector_tool import vector_search  # your RAG search function

app = FastAPI()

@app.get("/search")
def search(query: str):
    return {"result": vector_search(query)}
