from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagPaths:
    docs_dir: Path
    vectorstore_dir: Path


def get_paths() -> RagPaths:
    repo_root = Path(__file__).resolve().parents[1]
    return RagPaths(docs_dir=repo_root / "docs", vectorstore_dir=repo_root / "vectorstore")


CHROMA_COLLECTION = "lab_rag"
TOP_K = 3
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
CHAT_MODEL = "gpt-4o-mini"

