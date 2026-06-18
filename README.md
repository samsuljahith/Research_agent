# Research Agent — MCP + RAG + LangGraph

A research assistant that combines a local vector knowledge base with live web search fallback. Built on a client–server MCP architecture: the server exposes RAG tools over Qdrant, and the client runs a LangGraph ReAct agent via Groq.

## Architecture

```
Client (LangGraph ReAct Agent)
  └─ MCP Client → Server (FastAPI + MCP)
                    ├─ vector_search  →  Qdrant (local vector DB)
                    │                    └─ HuggingFace embeddings
                    └─ live_web_search  → web fallback if no match found
```

When a query has no match in the local knowledge base (similarity threshold not met), the agent automatically falls back to live web search.

## Components

| File | Role |
|------|------|
| `server/mcp_server.py` | FastAPI app exposing MCP-decorated tools |
| `server/vector_tool.py` | `vector_search` — Qdrant RAG with HF embeddings + web fallback |
| `server/search_tool.py` | `live_web_search` — web search tool |
| `server/data_loader.py` | Loads documents into Qdrant |
| `client/{agent_runner.py}` | LangGraph ReAct agent, MCP client, Groq LLM |

## Setup

### 1. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install & configure

```bash
pip install fastapi langchain-mcp-adapters langgraph langchain-groq qdrant-client llama-index-embeddings-huggingface python-dotenv

# .env
GROQ_API_KEY=your_key
```

### 3. Load documents

```bash
python server/data_loader.py   # indexes docs into Qdrant
```

### 4. Run the MCP server

```bash
uvicorn server.mcp_server:app --reload
```

### 5. Run the agent

```bash
python "client/{agent_runner.py}"
```

## Tech Stack

- **LLM** — Groq Mistral `mistral-saba-24b`
- **Agent** — LangGraph ReAct
- **Vector DB** — Qdrant
- **Embeddings** — HuggingFace `all-MiniLM-L6-v2`
- **MCP** — FastAPI + custom `@tool` decorator
