import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from rich.console import Console

from smartdoc.local_index import load_index
from smartdoc.settings import get_settings

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Smart Document Assistant.")
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
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    questions_path = Path("eval/questions.json")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_questions = len(questions)
    hit_count = 0

    for i, q_data in enumerate(questions):
        question = q_data["question"]
        expected_keywords = [kw.lower() for kw in q_data["expected_keywords"]]

        console.print(f"\n[bold]Question {i+1}:[/bold] {question}")

        q_vec = embeddings.embed_query(question)
        hits = index.search(np.array(q_vec, dtype="float32"), k=args.retrieve_chunks)
        retrieved_docs = [d for (d, _score) in hits]

        pairs = [[question, doc.page_content] for doc in retrieved_docs]
        scores = cross_encoder_model.predict(pairs)

        reranked_zipped_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs_with_scores = reranked_zipped_docs[:args.k]
        docs = [doc for doc, _score in reranked_docs_with_scores]

        console.print("[bold]Reranker Scores (Top 5):[/bold]")
        for j, (doc, score) in enumerate(reranked_docs_with_scores, start=1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", None)
            loc = f"{src}" + (f" (page {page})" if page is not None else "")
            console.print(f"  [{j}] {loc} - Score: {score:.4f}")

        found_keywords = []
        missing_keywords = []
        context_content = " ".join([doc.page_content.lower() for doc in docs])

        for keyword in expected_keywords:
            if keyword in context_content:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        if not missing_keywords:
            console.print(f"✅ All expected keywords found: {', '.join(found_keywords)}")
            hit_count += 1
        else:
            console.print(f"❌ Missing keywords: {', '.join(missing_keywords)}")
            if found_keywords:
                console.print(f"   Found keywords: {', '.join(found_keywords)}")
        
    hit_at_k = (hit_count / total_questions) * 100 if total_questions > 0 else 0
    console.print(f"\n[bold]Final Hit@k (k={args.k}) Score:[/bold] {hit_at_k:.2f}%")

if __name__ == "__main__":
    main()
