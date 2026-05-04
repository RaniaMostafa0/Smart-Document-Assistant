📄 Smart Document Assistant
A local RAG (Retrieval-Augmented Generation) system that lets you chat with your own documents using free Hugging Face models — no OpenAI API needed.
Built with LangChain for document loading and text splitting, FAISS for vector search, and Hugging Face for embeddings and LLM inference.

✨ Features

📁 Ingest PDFs, .txt, and .md files using LangChain document loaders
✂️ Smart text splitting with LangChain's RecursiveCharacterTextSplitter
🔍 Semantic search with FAISS + cosine similarity
🔁 Automatic query rewriting for better retrieval
🏆 Cross-encoder reranking for higher answer quality
💬 Two interfaces: CLI chat and Streamlit web UI
📊 Built-in evaluation script (Hit@k metric)
🆓 100% free — powered by Hugging Face Inference API


🧰 Tech Stack
ComponentLibraryDocument loadinglangchain-community (PyPDFLoader, TextLoader)Text splittinglangchain (RecursiveCharacterTextSplitter)Embeddingslangchain-huggingface + sentence-transformersVector storefaiss-cpuRerankingsentence-transformers CrossEncoderLLM inferencehuggingface-hub InferenceClientWeb UIstreamlitCLI outputrich

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/RaniaMostafa0/Smart-Document-Assistant.git
cd Smart-Document-Assistant
2. Create a virtual environment
bashpython -m venv .venv
Activate it:

Windows: .venv\Scripts\activate
Mac/Linux: source .venv/bin/activate

3. Install dependencies
bashpip install -r requirements.txt
4. Configure environment variables
bashcp .env.example .env
Edit .env and add your Hugging Face token:
envHUGGINGFACEHUB_API_TOKEN=your_token_here
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
Get your free token at huggingface.co/settings/tokens

📂 Add Your Documents
Place your PDF, .txt, or .md files in the data/ folder:
data/
├── document1.pdf
├── notes.txt
└── report.md

⚙️ Ingest Documents
Build the vector index from your documents:
bashpython -m smartdoc.ingest --data_dir data --index_dir index
LangChain loads and splits your documents into 1000-character chunks with 150-character overlap, then embeds them into a local FAISS index. This only needs to be run once, or whenever you add new documents.

💬 Chat with Your Documents
Option A — Streamlit Web UI (recommended):
bashpython -m streamlit run streamlit_app.py
Option B — CLI:
bashpython -m smartdoc.chat --index_dir index
Type your question and press Enter. Type exit to quit.

📊 Evaluate Retrieval Quality
Create eval/questions.json:
json[
  {
    "question": "What is the refund policy?",
    "expected_keywords": ["refund", "30 days"]
  }
]
Run the evaluation:
bashpython -m smartdoc.eval --index_dir index --k 5
This reports a Hit@k score — the percentage of questions where all expected keywords appear in the top-k retrieved chunks.

🛠️ Configuration Options
FlagDefaultDescription--k5Final number of chunks returned after reranking--retrieve_chunks20Candidate chunks fetched before reranking--data_dirdataFolder containing source documents--index_dirindexWhere the vector index is saved/loaded

🧠 How It Works
Your Question
     │
     ▼
Query Rewriting (Hugging Face LLM)
     │
     ▼
Vector Search (FAISS + cosine similarity)
     │
     ▼
Reranking (Cross-Encoder)
     │
     ▼
Answer Generation (Hugging Face LLM) → Cited Answer

Ingest — LangChain loaders read your files, RecursiveCharacterTextSplitter chunks them into 1000-character pieces, HuggingFaceEmbeddings embeds them, and they're stored in a local FAISS index
Query Rewrite — Your question is rewritten by an LLM for better search performance
Retrieve — Top 20 candidate chunks are fetched by cosine similarity
Rerank — A cross-encoder (ms-marco-MiniLM-L-6-v2) reranks candidates and selects the top 5
Answer — The LLM generates a grounded, cited answer using only the retrieved context


📁 Project Structure
Smart-Document-Assistant/
├── smartdoc/
│   ├── chat.py          # CLI chat interface
│   ├── ingest.py        # LangChain ingestion pipeline
│   ├── eval.py          # Retrieval evaluation
│   ├── local_index.py   # FAISS vector index
│   └── settings.py      # Environment configuration
├── streamlit_app.py     # Streamlit web UI
├── data/                # Your documents go here
├── eval/                # Evaluation questions
├── .env.example         # Environment template
└── requirements.txt

⚠️ Notes

The free Hugging Face Inference API has rate limits — responses may be slow at peak times
If a model is unavailable, the system automatically falls back to alternative models
The index/ folder is not included — run ingest after cloning
