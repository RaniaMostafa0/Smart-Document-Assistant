from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartdoc.local_index import LocalVectorIndex, save_index
from smartdoc.settings import get_settings


def _discover_pdfs(data_dir: Path) -> list[Path]:
    # One entry per file; resolve() avoids duplicate casing/short paths on Windows.
    by_key: dict[str, Path] = {}
    for p in data_dir.glob("**/*.pdf"):
        by_key[str(p.resolve())] = p
    return sorted(by_key.values(), key=lambda x: x.name.lower())


def _load_pdf_documents(data_dir: Path) -> list:
    paths = _discover_pdfs(data_dir)
    print(f"PDFs found under {data_dir}: {len(paths)}")
    for p in paths:
        print(f"  - {p}")

    docs = []
    for p in paths:
        try:
            part = PyPDFLoader(str(p)).load()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {p}") from e
        if not part:
            print(f"  (warning) no pages extracted: {p}")
        docs.extend(part)
    return docs


def _load_text_documents(data_dir: Path) -> list:
    # Brace globs like "**/*.{txt,md}" are not reliable with Path.glob / DirectoryLoader.
    loaders = []
    for pattern in ("**/*.txt", "**/*.md"):
        loaders.append(
            DirectoryLoader(
                str(data_dir),
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=False,
                use_multithreading=True,
            )
        )
    docs: list = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def build_vectorstore(data_dir: Path, index_dir: Path) -> None:
    settings = get_settings()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    docs: list = []
    docs.extend(_load_pdf_documents(data_dir))
    docs.extend(_load_text_documents(data_dir))

    if not docs:
        raise RuntimeError(
            f"No documents found in {data_dir}. Add PDFs or .txt/.md files and retry."
        )

    by_source = Counter(d.metadata.get("source", "?") for d in docs)
    print("Loaded document pages/chunks (pre-split) by file:")
    for src, n in sorted(by_source.items(), key=lambda x: str(x[0]).lower()):
        print(f"  {n:4d}  {src}")

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

