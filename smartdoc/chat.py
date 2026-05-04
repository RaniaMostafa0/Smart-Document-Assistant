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

import numpy as np
from sentence_transformers import CrossEncoder


def format_sources(docs: List[Document]) -> str:
    def safe_console_text(text: str) -> str:
        # Avoid Windows cp1252 encoding crashes from PDF glyphs.
        return text.encode("cp1252", errors="replace").decode("cp1252")

    lines = []
    for i, d in enumerate(docs, start=1):
        src = Path(d.metadata.get("source", "unknown")).name
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
            # Some free-provider accounts don\'t support all models.
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
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieve_chunks", type=int, default=20)
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

    # Load the cross-encoder model outside the loop
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_and_rerank_docs(query_text: str):
        q_vec = embeddings.embed_query(query_text)
        hits = index.search(np.array(q_vec, dtype="float32"), k=args.retrieve_chunks)
        retrieved_docs = [d for (d, _score) in hits]
        pairs = [[query_text, doc.page_content] for doc in retrieved_docs]
        scores = cross_encoder_model.predict(pairs)
        reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        return reranked[: args.k]

    console.print(
        Panel.fit(
            "Smart Document Assistant\n"
            "Type a question. Type \'exit\' to quit.",
            title="Ready",
        )
    )

    while True:
        q = console.input("\n[bold]You[/bold]> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Initial query rewriting prompt
        rewrite_prompt_template = (
            "Rewrite this question to be more specific and search-friendly for a "
            "document retrieval system.\n\nQuestion: {question}\nRewritten Question:"
        )

        # First rewrite and retrieval attempt
        rewritten_q1 = call_hf_chat_with_fallback(
            hf_client, settings.hf_llm_model, rewrite_prompt_template.format(question=q)
        )
        # console.print(f"[dim]Rewritten Q1:[/dim] {rewritten_q1}")
        reranked_docs_with_scores1 = get_and_rerank_docs(rewritten_q1)
        best_reranker_score1 = (
            reranked_docs_with_scores1[0][1] if reranked_docs_with_scores1 else -100.0
        )

        # Initialize before the if block
        best_reranker_score2 = -100.0
        reranked_docs_with_scores2 = []

        # If top score is below threshold, try a second rewrite
        if best_reranker_score1 < -2.0:
            # console.print("[dim]Reranker confidence low, trying second rewrite...[/dim]")
            rewrite_prompt_template2 = (
                "The previous search was not confident. Rewrite this question again, "
                "focusing on a different angle or using alternative keywords, to be "
                "more specific and search-friendly for a document retrieval system.\n\n"
                "Original Question: {original_question}\n"
                "Previous Rewritten Question: {previous_rewrite}\n"
                "Rewritten Question (second attempt):"
            )
            rewritten_q2 = call_hf_chat_with_fallback(
                hf_client,
                settings.hf_llm_model,
                rewrite_prompt_template2.format(
                    original_question=q, previous_rewrite=rewritten_q1
                ),
            )
            # console.print(f"[dim]Rewritten Q2:[/dim] {rewritten_q2}")
            reranked_docs_with_scores2 = get_and_rerank_docs(rewritten_q2)
            best_reranker_score2 = (
                reranked_docs_with_scores2[0][1]
                if reranked_docs_with_scores2
                else -100.0
            )

            if best_reranker_score2 > best_reranker_score1:
                # console.print("[dim]Second rewrite performed better.[/dim]")
                pass
        
        # Use whichever attempt was better
        if best_reranker_score1 < -2.0 and best_reranker_score2 > best_reranker_score1:
            final_reranked_docs_with_scores = reranked_docs_with_scores2
        else:
            final_reranked_docs_with_scores = reranked_docs_with_scores1

        docs = [doc for doc, _score in final_reranked_docs_with_scores]

        # console.print("\n[bold]Reranker Scores:[/bold]")
        # for i, (doc, score) in enumerate(final_reranked_docs_with_scores, start=1):
        #     src = doc.metadata.get("source", "unknown")
        #     page = doc.metadata.get("page", None)
        #     loc = f"{src}" + (f" (page {page})" if page is not None else "")
        #     console.print(f"[{i}] {loc} - Score: {score:.4f}")

        context = "\n\n".join(
            f"Source {i}:\n{d.page_content}" for i, d in enumerate(docs, start=1)
        )

        prompt_template = (
            "You are a helpful assistant. Answer the user using ONLY the context.\n"
            "If the answer is not in the context, say you don\'t know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer (be concise, include source numbers like [1][2] when relevant):"
        )
        prompt = prompt_template.format(context=context, question=q)

        answer = call_hf_chat_with_fallback(hf_client, settings.hf_llm_model, prompt)
        console.print(Panel(answer.strip(), title="Answer"))
        console.print(Panel(format_sources(docs), title="Sources"))


if __name__ == "__main__":
    main()
