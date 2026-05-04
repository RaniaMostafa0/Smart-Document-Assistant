from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document


@dataclass(frozen=True)
class LocalVectorIndex:
    embeddings: np.ndarray  # shape: (n, d), float32
    documents: List[Document]

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        if self.embeddings.size == 0:
            return []
        k = max(1, min(int(k), self.embeddings.shape[0]))

        # FAISS uses inner product here; normalize vectors to approximate cosine similarity.
        docs_mat = self.embeddings.astype("float32", copy=False)
        docs_norm = np.linalg.norm(docs_mat, axis=1, keepdims=True)
        docs_norm = np.maximum(docs_norm, 1e-12)
        docs_unit = docs_mat / docs_norm

        q = query_vec.reshape(1, -1).astype("float32", copy=False)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.maximum(q_norm, 1e-12)
        q_unit = q / q_norm

        index = faiss.IndexFlatIP(docs_unit.shape[1])
        index.add(docs_unit)
        scores, indices = index.search(q_unit, k)

        out: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
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

