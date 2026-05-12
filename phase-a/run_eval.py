from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Allow running as `python phase-a/run_eval.py` while importing repo-root modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from your_rag_module import my_rag_pipeline
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment (or copy `.env.example` to `.env`) "
            "before running evaluation."
        )


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid threshold '{item}'. Use name=value, e.g. faithfulness=0.85")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in METRIC_NAMES:
            raise ValueError(f"Unknown metric '{name}'. Valid metrics: {', '.join(METRIC_NAMES)}")
        thresholds[name] = float(value.strip())
    return thresholds


def _coerce_contexts(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # testset generator stores JSON in CSV
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [s]
    return [str(x)]


def _run_rag(question: str):
    out = my_rag_pipeline(question)
    if isinstance(out, dict):
        answer = out.get("answer", "")
        contexts = out.get("contexts", [])
        return str(answer), _coerce_contexts(contexts)
    if isinstance(out, tuple) and len(out) == 2:
        answer, contexts = out
        return str(answer), _coerce_contexts(contexts)
    raise RuntimeError("my_rag_pipeline returned an unsupported type; expected (answer, contexts) or dict.")


def _load_testset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"Empty testset: {path}")
    for col in ["question", "ground_truth", "contexts", "evolution_type"]:
        if col not in (rows[0].keys() if rows else []):
            raise RuntimeError(f"Missing column '{col}' in {path}")
    return rows


def _ragas_evaluate(dataset: Dataset) -> pd.DataFrame:
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to import RAGAS metrics ({e}). Ensure `ragas` is installed.") from e

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # RAGAS API differs across versions; try common calling conventions.
    try:
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    except TypeError:
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if isinstance(result, pd.DataFrame):
        return result

    raise RuntimeError("Unexpected RAGAS evaluate() return type; expected something convertible to pandas.")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Run RAG pipeline on testset and evaluate with RAGAS.")
    parser.add_argument(
        "--threshold",
        nargs="*",
        default=[],
        help=(
            "Metric thresholds as name=value pairs. Example: "
            "--threshold faithfulness=0.85 answer_relevancy=0.80 context_precision=0.70 context_recall=0.75"
        ),
    )
    args = parser.parse_args(argv)
    try:
        thresholds = _parse_thresholds(args.threshold)
    except Exception as e:
        print(f"ERROR: {e}")
        return 2

    try:
        _require_openai_api_key()
        testset_path = Path("phase-a") / "testset_v1.csv"
        if not testset_path.exists():
            raise RuntimeError(f"Missing testset file: {testset_path}")

        test_rows = _load_testset(testset_path)

        results_data: list[dict[str, Any]] = []
        for r in test_rows:
            q = (r.get("question") or "").strip()
            if not q:
                continue

            answer, contexts = _run_rag(q)
            ground_truth = (r.get("ground_truth") or "").strip()

            results_data.append(
                {"question": q, "answer": answer, "contexts": contexts, "ground_truth": ground_truth}
            )

        if not results_data:
            raise RuntimeError("No valid questions found in phase-a/testset_v1.csv")

        # RAGAS expects: question, answer, contexts (list[str]), ground_truth
        ds = Dataset.from_list(results_data)
        ragas_df = _ragas_evaluate(ds)

        # Persist row-level results
        out_results = Path("phase-a") / "ragas_results.csv"
        merged = pd.concat([pd.DataFrame(results_data), ragas_df.reset_index(drop=True)], axis=1)
        merged.to_csv(out_results, index=False)

        # Aggregate
        agg: dict[str, float] = {}
        for m in METRIC_NAMES:
            if m in merged.columns:
                agg[m] = float(pd.to_numeric(merged[m], errors="coerce").mean())

        out_summary = Path("phase-a") / "ragas_summary.json"
        summary = {"metrics": agg, "thresholds": thresholds}
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Gate
        failed: list[str] = []
        for name, thr in thresholds.items():
            score = agg.get(name)
            if score is None:
                failed.append(f"{name}=NA (< {thr})")
            elif score < thr:
                failed.append(f"{name}={score:.3f} (< {thr})")

        # Print concise summary
        printable = "  ".join([f"{k}={v:.3f}" for k, v in agg.items()]) if agg else "no metrics computed"
        print(f"RAGAS summary: {printable}")
        print(f"Wrote: {out_results}")
        print(f"Wrote: {out_summary}")

        if failed:
            print("THRESHOLD FAIL: " + "; ".join(failed))
            return 1
        print("THRESHOLD PASS")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
