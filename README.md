# Smart Document Assistant (LangChain → LangGraph)

This project is a learning-first **Document Q&A (RAG)** assistant.

## What you’ll build first (LangChain)
- Load local documents (PDF/TXT/MD)
- Split into chunks
- Create embeddings (Hugging Face)
- Store/search chunks (local index)
- Ask questions with a Hugging Face LLM endpoint + retrieved context

## Setup
### Python version (important)
Use **Python 3.12** (or 3.11). Python 3.14 on Windows currently ships an experimental NumPy build that may crash when running ML/NLP libraries.

1) Create a virtualenv and install deps

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Add your Hugging Face token

Copy `.env.example` to `.env` and fill `HUGGINGFACEHUB_API_TOKEN`.

## Put documents here
- `data/` (you can add PDFs, .txt, .md)

## Build the index

```bash
python -m smartdoc.ingest --data_dir data --index_dir index
```

## Chat with your documents

```bash
python -m smartdoc.chat --index_dir index
```

## Next step (LangGraph)
Once the LangChain version is solid, we’ll refactor the chat flow into a LangGraph:
- retrieval node
- answer node
- memory/checkpointing
- tool routing (optional)

