from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    get_paths,
)


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment before building the index "
            "(e.g., `setx OPENAI_API_KEY \"...\"` on Windows or export it in your shell)."
        )


def _list_markdown_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    return sorted([p for p in docs_dir.rglob("*.md") if p.is_file()])


def build_index() -> None:
    paths = get_paths()
    md_files = _list_markdown_files(paths.docs_dir)
    if not md_files:
        raise RuntimeError("No markdown files found in ./docs. Add at least one `*.md` file to index.")

    _require_openai_api_key()

    loader = DirectoryLoader(
        str(paths.docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(documents)

    paths.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(paths.vectorstore_dir),
    )
    # For compatibility across Chroma/LangChain versions:
    if hasattr(vectordb, "persist"):
        vectordb.persist()


def main() -> int:
    try:
        build_index()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    print("Index built and persisted to ./vectorstore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
