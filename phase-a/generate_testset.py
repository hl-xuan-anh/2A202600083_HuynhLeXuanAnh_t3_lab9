from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment "
            "(or copy `.env.example` to `.env`) before generating the test set."
        )


def _require_docs() -> None:
    docs_dir = Path("docs")
    md_files = list(docs_dir.rglob("*.md")) if docs_dir.exists() else []
    if not md_files:
        raise RuntimeError("No markdown files found in ./docs. Add at least one `*.md` file first.")


def _load_docs():
    loader = DirectoryLoader(
        "./docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["doc_id"] = f"doc_{i}"
        doc.metadata["filename"] = doc.metadata.get("source", f"doc_{i}")

    return docs


def _make_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        timeout=120,
        max_retries=5,
    )


def _make_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        timeout=120,
        max_retries=5,
        chunk_size=1,
    )


def _get_ragas_generator():
    try:
        from ragas.testset.generator import TestsetGenerator  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to import RAGAS TestsetGenerator ({e}). Ensure `ragas` is installed."
        ) from e

    try:
        from ragas.testset.evolutions import simple, reasoning, multi_context  # type: ignore

        evolutions = {
            "simple": simple,
            "reasoning": reasoning,
            "multi_context": multi_context,
        }
    except Exception:
        evolutions = {}

    llm = _make_llm()
    embeddings = _make_embeddings()

    try:
        generator = TestsetGenerator.from_langchain(
            generator_llm=llm,
            critic_llm=llm,
            embeddings=embeddings,
        )
    except TypeError:
        generator = TestsetGenerator.from_langchain(
            llm,
            llm,
            embeddings,
        )

    return generator, evolutions


def _make_run_config():
    try:
        from ragas.run_config import RunConfig  # type: ignore

        try:
            return RunConfig(
                timeout=120,
                max_retries=5,
                max_wait=60,
                max_workers=2,
                max_concurrent=2,
            )
        except TypeError:
            return RunConfig(
                timeout=120,
                max_retries=5,
                max_wait=60,
                max_workers=2,
            )

    except Exception:
        return None


def _generate_testset(generator, docs, test_size: int, distributions, dist_payload):
    run_config = _make_run_config()

    kwargs = {
        "test_size": test_size,
        "distributions": dist_payload if dist_payload else distributions,
    }

    if run_config is not None:
        kwargs["run_config"] = run_config

    try:
        return generator.generate_with_langchain_docs(docs, **kwargs)
    except TypeError:
        kwargs.pop("run_config", None)
        return generator.generate_with_langchain_docs(docs, **kwargs)


def _to_csv_frame(testset) -> pd.DataFrame:
    if hasattr(testset, "to_pandas"):
        df = testset.to_pandas()
    elif isinstance(testset, pd.DataFrame):
        df = testset
    else:
        df = pd.DataFrame(list(testset))

    rename = {}

    if "answer" in df.columns and "ground_truth" not in df.columns:
        rename["answer"] = "ground_truth"

    if "ground_truths" in df.columns and "ground_truth" not in df.columns:
        rename["ground_truths"] = "ground_truth"

    if "question" not in df.columns and "query" in df.columns:
        rename["query"] = "question"

    if "contexts" not in df.columns and "context" in df.columns:
        rename["context"] = "contexts"

    df = df.rename(columns=rename)

    for col in ["question", "ground_truth", "contexts", "evolution_type"]:
        if col not in df.columns:
            df[col] = ""

    def _serialize_contexts(x):
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    df["contexts"] = df["contexts"].apply(_serialize_contexts)

    return df[["question", "ground_truth", "contexts", "evolution_type"]]


def main() -> int:
    load_dotenv(override=False)

    try:
        _require_openai_api_key()
        _require_docs()

        docs = _load_docs()
        generator, evolutions = _get_ragas_generator()

        distributions = {
            "simple": 0.5,
            "reasoning": 0.25,
            "multi_context": 0.25,
        }

        dist_payload = {}
        if evolutions:
            dist_payload = {
                evolutions[k]: v
                for k, v in distributions.items()
                if k in evolutions
            }

        testset = _generate_testset(
            generator=generator,
            docs=docs,
            test_size=30,
            distributions=distributions,
            dist_payload=dist_payload,
        )

        df = _to_csv_frame(testset)

        out_path = Path("phase-a") / "testset_v1.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")

        print(f"Wrote test set to {out_path}")
        print("\nShape:")
        print(df.shape)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nEvolution type counts:")
        print(df["evolution_type"].value_counts(dropna=False))

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())