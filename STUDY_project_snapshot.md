# Smart Document Assistant — study snapshot

**Purpose:** Single file to read the whole mini-project offline.  
**Canonical code:** The runnable app lives under `smartdoc/`; this document is **not** executed by Python—only for study.

---

## Contents

1. [How modules connect](#how-modules-connect)
2. [requirements.txt](#requirementstxt)
3. [README.md](#readmemd-copy)
4. [.env.example](#envexample-copy)
5. [`smartdoc/__init__.py`](#smartdoc__init__py)
6. [`smartdoc/__main__.py`](#smartdoc__main__py)
7. [`smartdoc/settings.py`](#smartdoc_settingspy)
8. [`smartdoc/local_index.py`](#smartdoc_local_indexpy)
9. [`smartdoc/ingest.py`](#smartdoc_ingestpy)
10. [`smartdoc/chat.py`](#smartdoc_chatpy)

---

## How modules connect

- **`settings.get_settings()`** — loads `.env` (HF token + optional model IDs).
- **`ingest`** — loads files from `data/`, splits, embeds, builds `LocalVectorIndex`, **`save_index`** → `index/index.pkl`.
- **`chat`** — **`load_index`**, embeds the user question, **`index.search`**, builds RAG prompt, calls **Hugging Face Inference API**.
- **`__main__`** — prints CLI hints only.

---

## requirements.txt

```text
langchain>=0.2.16
langchain-community>=0.2.16
langchain-huggingface>=0.0.3
python-dotenv>=1.0.1
sentence-transformers>=3.0.1
pypdf>=4.2.0
rich>=13.7.1
scikit-learn>=1.8.0
```

---

## README.md (copy)

```text
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

    python -m venv .venv
    # Windows PowerShell:
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

2) Add your Hugging Face token

Copy `.env.example` to `.env` and fill `HUGGINGFACEHUB_API_TOKEN`.

## Put documents here
- `data/` (you can add PDFs, .txt, .md)

## Build the index

    python -m smartdoc.ingest --data_dir data --index_dir index

## Chat with your documents

    python -m smartdoc.chat --index_dir index

## Next step (LangGraph)
Once the LangChain version is solid, we’ll refactor the chat flow into a LangGraph:
- retrieval node
- answer node
- memory/checkpointing
- tool routing (optional)
```

---

## .env.example (copy)

> Keep real secrets in `.gitignore`’d `.env`. Replace the placeholder token in `.env.example` locally if yours is committed by mistake—rotate tokens if exposed.

```
HUGGINGFACEHUB_API_TOKEN=your_token_here

# Choose an LLM hosted on Hugging Face (works via Inference API).
# Good defaults (you can change later):
# - Qwen/Qwen2.5-7B-Instruct
# - HuggingFaceH4/zephyr-7b-beta
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Embedding model for local embeddings (downloaded once).
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

*(For an exact replica of `.env.example` on disk—including its current token line—open `smartdoc/../.env.example` in the repo; this study file uses a neutral placeholder.)*

---

## smartdoc/__init__.py

```python
__all__ = []

```

---

## smartdoc/__main__.py

```python
from __future__ import annotations

import sys


def main() -> None:
    print(
        "Use one of:\n"
        "  python -m smartdoc.ingest --data_dir data --index_dir index\n"
        "  python -m smartdoc.chat --index_dir index\n"
    )


if __name__ == "__main__":
    sys.exit(main())

```

---

## smartdoc/settings.py

```python
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    huggingface_token: str
    hf_llm_model: str
    hf_embed_model: str


def get_settings() -> Settings:
    load_dotenv()

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

    return Settings(
        huggingface_token=token,
        hf_llm_model=os.getenv(
            "HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"
        ).strip(),
        hf_embed_model=os.getenv(
            "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
    )

```

---

## smartdoc/local_index.py

```python
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LocalVectorIndex:
    embeddings: np.ndarray  # shape: (n, d), float32
    documents: List[Document]

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        if self.embeddings.size == 0:
            return []
        k = max(1, min(int(k), self.embeddings.shape[0]))

        # cosine distance = 1 - cosine similarity
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(self.embeddings)
        distances, indices = nn.kneighbors(query_vec.reshape(1, -1), n_neighbors=k)

        out: List[Tuple[Document, float]] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            score = 1.0 - float(dist)  # similarity
            out.append((self.documents[idx], score))
        return out


def save_index(index: LocalVectorIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / "index.pkl"
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(index_dir: Path) -> LocalVectorIndex:
    path = index_dir / "index.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}. Run ingest first.")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, LocalVectorIndex):
        raise TypeError("index.pkl is not a LocalVectorIndex (wrong file?)")
    return obj

```

---

## smartdoc/ingest.py

```python
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartdoc.local_index import LocalVectorIndex, save_index
from smartdoc.settings import get_settings


def build_vectorstore(data_dir: Path, index_dir: Path) -> None:
    settings = get_settings()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    loaders = []
    # PDF
    loaders.append(
        DirectoryLoader(
            str(data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )
    )
    # Text-like
    loaders.append(
        DirectoryLoader(
            str(data_dir),
            glob="**/*.{txt,md}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
    )

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    if not docs:
        raise RuntimeError(
            f"No documents found in {data_dir}. Add PDFs or .txt/.md files and retry."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.hf_embed_model)

    import numpy as np

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    vecs = np.array(vectors, dtype="float32")

    save_index(LocalVectorIndex(embeddings=vecs, documents=chunks), index_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a local vector index."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--index_dir", type=str, default="index")
    args = parser.parse_args()

    build_vectorstore(Path(args.data_dir), Path(args.index_dir))
    print(f"Saved index to: {args.index_dir}")


if __name__ == "__main__":
    main()

```

---

## smartdoc/chat.py

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.panel import Panel

from smartdoc.local_index import load_index
from smartdoc.settings import get_settings


def format_sources(docs: List[Document]) -> str:
    def safe_console_text(text: str) -> str:
        # Avoid Windows cp1252 encoding crashes from PDF glyphs.
        return text.encode("cp1252", errors="replace").decode("cp1252")

    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        loc = f"{src}" + (f" (page {page})" if page is not None else "")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        lines.append(safe_console_text(f"[{i}] {loc}\n    {snippet}"))
    return "\n".join(lines) if lines else "(no sources)"


def call_hf_chat_with_fallback(
    client: InferenceClient, preferred_model: str, prompt: str
) -> str:
    candidate_models = [
        preferred_model,
        "Qwen/Qwen2.5-7B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
    ]

    last_error: Exception | None = None
    for model_id in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.2,
            )
            return (response.choices[0].message.content or "").strip()
        except BadRequestError as e:
            # Some free-provider accounts don't support all models.
            if "model_not_supported" in str(e):
                last_error = e
                continue
            raise

    raise RuntimeError(
        "No supported chat model found. Set HF_LLM_MODEL in .env to a model your "
        "Hugging Face account supports (e.g. Qwen/Qwen2.5-7B-Instruct)."
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your documents (RAG).")
    parser.add_argument("--index_dir", type=str, default="index")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    console = Console()
    if not settings.huggingface_token:
        raise RuntimeError(
            "Missing HUGGINGFACEHUB_API_TOKEN. Copy .env.example to .env and set it."
        )

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"index_dir not found: {index_dir}. Run ingest first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=settings.hf_embed_model)
    index = load_index(index_dir)

    hf_client = InferenceClient(token=settings.huggingface_token)

    console.print(
        Panel.fit(
            "Smart Document Assistant (LangChain RAG)\n"
            "Type a question. Type 'exit' to quit.",
            title="Ready",
        )
    )

    while True:
        q = console.input("\n[bold]You[/bold]> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        q_vec = embeddings.embed_query(q)
        import numpy as np

        hits = index.search(np.array(q_vec, dtype="float32"), k=args.k)
        docs = [d for (d, _score) in hits]
        context = "\n\n".join(
            f"Source {i}:\n{d.page_content}" for i, d in enumerate(docs, start=1)
        )

        prompt = (
            "You are a helpful assistant. Answer the user using ONLY the context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer (be concise, include source numbers like [1][2] when relevant):"
        )

        answer = call_hf_chat_with_fallback(hf_client, settings.hf_llm_model, prompt)
        console.print(Panel(answer.strip(), title="Answer"))
        console.print(Panel(format_sources(docs), title="Sources"))


if __name__ == "__main__":
    main()

```
