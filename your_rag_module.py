from __future__ import annotations

"""
Compatibility shim for Phase A evaluation.

The assignment spec expects:
    from your_rag_module import my_rag_pipeline

This repo's RAG implementation lives in `rag_pipeline/`.
"""

from rag_pipeline import my_rag_pipeline  # noqa: F401

