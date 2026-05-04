from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from smartdoc.local_index import load_index
from smartdoc.settings import get_settings


# ---------------------------
# Clean source formatting
# ---------------------------
def clean_source_name(path: str) -> str:
    name = Path(path).name
    name = name.replace("_", " ").replace(".pdf", "")
    return name


def format_sources(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = clean_source_name(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", None)
        loc = f"{src}" + (f" (p. {page})" if page is not None else "")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"[{i}] {loc}\n    {snippet}")
    return "\n\n".join(lines) if lines else "(no sources)"


# ---------------------------
# Cached resource loaders
# Loaded ONCE on startup, reused on every rerun — eliminates the delay
# ---------------------------
@st.cache_resource
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def load_vector_index(index_dir: str):
    return load_index(Path(index_dir))


@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_hf_client(token: str):
    return InferenceClient(token=token)


# ---------------------------
# HF LLM call
# ---------------------------
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
            if "model_not_supported" in str(e):
                last_error = e
                continue
            raise

    raise RuntimeError(
        "No supported chat model found. Set HF_LLM_MODEL in .env."
    ) from last_error


# ---------------------------
# Retrieval + reranking
# ---------------------------
def get_and_rerank_docs(
    query_text: str,
    embeddings,
    index,
    cross_encoder_model,
    k,
    retrieve_chunks,
):
    q_vec = embeddings.embed_query(query_text)
    hits = index.search(np.array(q_vec, dtype="float32"), k=retrieve_chunks)
    retrieved_docs = [d for (d, _score) in hits]

    pairs = [[query_text, doc.page_content] for doc in retrieved_docs]
    scores = cross_encoder_model.predict(pairs)

    reranked = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return reranked[:k]


# ---------------------------
# MAIN APP
# ---------------------------
def main() -> None:
    st.set_page_config(layout="wide")
    st.title("📄 Smart Document Assistant")

    settings = get_settings()
    if not settings.huggingface_token:
        st.error("Missing HUGGINGFACEHUB_API_TOKEN.")
        st.stop()

    if not Path("index").exists():
        st.error("Index not found. Run ingest first.")
        st.stop()

    # All heavy objects loaded once and cached — zero reload cost on reruns
    embeddings = load_embeddings(settings.hf_embed_model)
    index = load_vector_index("index")
    hf_client = load_hf_client(settings.huggingface_token)
    cross_encoder_model = load_cross_encoder()

    # Sidebar
    st.sidebar.header("Settings")
    k_value = st.sidebar.slider("Top K (final)", 1, 10, 5)
    retrieve_chunks_value = st.sidebar.slider("Retrieve candidates", 10, 50, 20)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources"):
                    st.write(msg["sources"])

    # Input
    q = st.chat_input("Ask a question about your documents...")

    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=True)

            # Query rewrite
            status.write("✏️ Rewriting your question for better search…")
            rewrite_prompt = (
                "Rewrite this question to be clearer and better for document search:\n\n"
                f"{q}"
            )
            rewritten_q = call_hf_chat_with_fallback(
                hf_client, settings.hf_llm_model, rewrite_prompt
            )

            # Retrieval
            status.write("🔍 Searching documents…")
            reranked_docs_with_scores = get_and_rerank_docs(
                rewritten_q,
                embeddings,
                index,
                cross_encoder_model,
                k_value,
                retrieve_chunks_value,
            )
            docs = [doc for doc, _ in reranked_docs_with_scores]

            # Build context
            context = "\n\n".join(
                f"Source {i}:\n{d.page_content}"
                for i, d in enumerate(docs, start=1)
            )
            prompt = (
                "You are a helpful assistant. Answer ONLY using the context.\n"
                "If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\n"
                "Answer (concise, cite sources like [1][2]):"
            )

            # LLM answer
            status.write("💬 Generating answer…")
            answer = call_hf_chat_with_fallback(
                hf_client, settings.hf_llm_model, prompt
            )

            status.update(label="Done!", state="complete", expanded=False)

            st.write(answer)
            sources_text = format_sources(docs)
            with st.expander("📚 Sources"):
                st.write(sources_text)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources_text}
        )


if __name__ == "__main__":
    main()