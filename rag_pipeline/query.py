from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import CHAT_MODEL, CHROMA_COLLECTION, TOP_K, get_paths


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment before querying "
            "(e.g., `setx OPENAI_API_KEY \"...\"` on Windows or export it in your shell)."
        )


def _list_markdown_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        return []
    return sorted([p for p in docs_dir.rglob("*.md") if p.is_file()])


def _require_docs_present() -> None:
    paths = get_paths()
    if not _list_markdown_files(paths.docs_dir):
        raise RuntimeError("No markdown files found in ./docs. Add at least one `*.md` file first.")


def _require_vectorstore_built() -> None:
    paths = get_paths()
    if not paths.vectorstore_dir.exists():
        raise RuntimeError(
            "Vectorstore folder missing. Build it first with: `python -m rag_pipeline.build_index` "
            "(this creates ./vectorstore)."
        )

    entries = [p for p in paths.vectorstore_dir.iterdir() if p.name != ".gitkeep"]
    if not entries:
        raise RuntimeError(
            "Vectorstore not found. Build it first with: `python -m rag_pipeline.build_index` "
            "(this creates ./vectorstore)."
        )


def _load_vectorstore() -> Chroma:
    paths = get_paths()
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(paths.vectorstore_dir),
        embedding_function=embeddings,
    )


def my_rag_pipeline(question: str) -> tuple[str, list[str]]:
    _require_openai_api_key()
    _require_docs_present()
    _require_vectorstore_built()

    vectordb = _load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.get_relevant_documents(question)
    contexts = [d.page_content for d in docs]

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided contexts. "
        "If the contexts are insufficient, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Contexts:\n{context_block}"
    )
    answer = llm.invoke([HumanMessage(content=prompt)]).content
    return str(answer).strip(), contexts


async def my_rag_pipeline_async(question: str) -> tuple[str, list[str]]:
    _require_openai_api_key()
    _require_docs_present()
    _require_vectorstore_built()

    vectordb = _load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # Some LangChain versions support async retrieval; keep robust fallback.
    if hasattr(retriever, "aget_relevant_documents"):
        docs = await retriever.aget_relevant_documents(question)  # type: ignore[attr-defined]
    else:
        docs = retriever.get_relevant_documents(question)

    contexts = [d.page_content for d in docs]
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided contexts. "
        "If the contexts are insufficient, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Contexts:\n{context_block}"
    )

    if hasattr(llm, "ainvoke"):
        resp = await llm.ainvoke([HumanMessage(content=prompt)])  # type: ignore[call-arg]
        answer = getattr(resp, "content", resp)
    else:
        answer = llm.invoke([HumanMessage(content=prompt)]).content
    return str(answer).strip(), contexts


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print('Usage: python -m rag_pipeline.query "What is X?"')
        return 2
    question = " ".join(argv).strip()
    try:
        answer, contexts = my_rag_pipeline(question)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(answer)
    print("\nContexts:")
    for i, ctx in enumerate(contexts, start=1):
        print(f"\n[{i}]\n{ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
