from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass(frozen=True)
class RowView:
    question: str
    evolution_type: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    avg: float | None
    cluster: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle cases where ragas_results.csv accidentally contains duplicate columns
    (e.g., question,answer,contexts repeated).
    Keep the first occurrence.
    """
    cols = list(df.columns)
    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, c in enumerate(cols):
        if c in seen:
            continue
        keep_idx.append(i)
        seen.add(c)
    return df.iloc[:, keep_idx]


def _coerce_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df))
    return pd.to_numeric(df[col], errors="coerce")


def _truncate(s: str, max_len: int = 80) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _load_evolution_map(testset_path: Path) -> dict[str, str]:
    """
    Map question -> evolution_type when possible.
    """
    if not testset_path.exists():
        return {}
    try:
        df = pd.read_csv(testset_path)
    except Exception:
        return {}
    if "question" not in df.columns:
        return {}
    evo_col = "evolution_type" if "evolution_type" in df.columns else None
    if evo_col is None:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        q = str(r.get("question", "")).strip()
        if not q:
            continue
        out[q] = str(r.get(evo_col, "")).strip()
    return out


def _assign_cluster(
    evolution_type: str,
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_precision: float | None,
    context_recall: float | None,
) -> str:
    evo = (evolution_type or "").lower()

    def low(x: float | None, thr: float) -> bool:
        return x is not None and pd.notna(x) and float(x) < thr

    # C4: multi-hop/reasoning (label-driven) — call it out early for visibility.
    if "reasoning" in evo or "multi_context" in evo or "multi-context" in evo:
        return "C4 multi-hop/reasoning"

    # C1: retrieval failure when context metrics are low.
    if low(context_precision, 0.70) or low(context_recall, 0.70):
        return "C1 retrieval failure"

    # C2: hallucination/faithfulness.
    if low(faithfulness, 0.75):
        return "C2 faithfulness/hallucination"

    # C3: answer relevance.
    if low(answer_relevancy, 0.75):
        return "C3 answer relevance"

    return "C0 other"


def _compute_avg(row: dict[str, float | None]) -> float | None:
    vals: list[float] = []
    for m in METRICS:
        v = row.get(m)
        if v is None or pd.isna(v):
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _make_markdown_table(rows: list[RowView]) -> str:
    header = "| # | Question | Type | F | AR | CP | CR | Avg | Cluster |\n|---:|---|---|---:|---:|---:|---:|---:|---|\n"

    def fmt(x: float | None) -> str:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.3f}"

    lines = []
    for i, r in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    _truncate(r.question, 80).replace("|", "\\|"),
                    (r.evolution_type or "NA").replace("|", "\\|"),
                    fmt(r.faithfulness),
                    fmt(r.answer_relevancy),
                    fmt(r.context_precision),
                    fmt(r.context_recall),
                    fmt(r.avg),
                    r.cluster.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return header + "\n".join(lines) + "\n"


def _cluster_sections(rows: list[RowView]) -> str:
    """
    Produce at least 2 cluster sections based on the most frequent clusters
    among the bottom rows (excluding C0).
    """
    counts = Counter([r.cluster for r in rows if not r.cluster.startswith("C0")])
    top = [c for c, _n in counts.most_common(3)]
    if len(top) < 2:
        # Ensure at least 2 sections
        all_clusters = [c for c in ["C1 retrieval failure", "C2 faithfulness/hallucination", "C3 answer relevance", "C4 multi-hop/reasoning"] if c in counts or True]
        top = (top + [c for c in all_clusters if c not in top])[:2]

    def examples_for(cluster: str) -> list[str]:
        ex = [r.question for r in rows if r.cluster == cluster][:3]
        return [_truncate(x, 120) for x in ex] if ex else []

    sections: list[str] = []
    for cluster in top[:2]:
        examples = examples_for(cluster)
        sections.append(f"## {cluster}\n")
        if cluster.startswith("C1"):
            sections.append("**Pattern:** Low context precision/recall suggests retrieval is missing or noisy.\n")
            sections.append("**Examples:**\n" + ("\n".join([f"- {e}" for e in examples]) or "- (no examples found)") + "\n")
            sections.append("**Root cause:** Chunking or embeddings/indexing may not align with the questions; top_k too small; missing metadata filtering.\n")
            sections.append("**Proposed technical fix:** Improve chunking (size/overlap), add hybrid search or reranking, increase `top_k`, and ensure `./docs` coverage matches the testset.\n")
        elif cluster.startswith("C2"):
            sections.append("**Pattern:** Faithfulness low even when some context exists → model may be inventing details not supported by retrieved text.\n")
            sections.append("**Examples:**\n" + ("\n".join([f"- {e}" for e in examples]) or "- (no examples found)") + "\n")
            sections.append("**Root cause:** Prompt does not strongly constrain the model; retrieved contexts too long/irrelevant; missing citations requirement.\n")
            sections.append("**Proposed technical fix:** Tighten prompt (\"use only contexts\" + refusal), trim/format contexts, and add a post-check to refuse when evidence is weak.\n")
        elif cluster.startswith("C3"):
            sections.append("**Pattern:** Answer relevancy low → answer drifts from the asked question.\n")
            sections.append("**Examples:**\n" + ("\n".join([f"- {e}" for e in examples]) or "- (no examples found)") + "\n")
            sections.append("**Root cause:** Retrieval brings generic content; prompt lacks direct instruction to answer succinctly and directly.\n")
            sections.append("**Proposed technical fix:** Add query rewriting, stronger question-focused prompt, and stricter selection of contexts.\n")
        elif cluster.startswith("C4"):
            sections.append("**Pattern:** Multi-context/reasoning questions often need evidence across multiple chunks.\n")
            sections.append("**Examples:**\n" + ("\n".join([f"- {e}" for e in examples]) or "- (no examples found)") + "\n")
            sections.append("**Root cause:** Single-pass retrieval misses supporting steps; contexts are fragmented.\n")
            sections.append("**Proposed technical fix:** Increase `top_k`, add multi-query retrieval, or iterative retrieval (retrieve → draft → retrieve again).\n")
        else:
            sections.append("**Pattern:** Mixed or unknown.\n")
            sections.append("**Examples:**\n" + ("\n".join([f"- {e}" for e in examples]) or "- (no examples found)") + "\n")
            sections.append("**Root cause:** Unknown.\n")
            sections.append("**Proposed technical fix:** Inspect the bottom examples and add targeted eval cases.\n")
        sections.append("\n")

    return "".join(sections)


def main() -> int:
    results_path = Path("phase-a") / "ragas_results.csv"
    testset_path = Path("phase-a") / "testset_v1.csv"
    out_md = Path("phase-a") / "failure_analysis.md"

    try:
        df = _dedupe_columns(_read_csv(results_path))
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    evo_map = _load_evolution_map(testset_path)

    if "question" not in df.columns:
        print("ERROR: ragas_results.csv missing required column 'question'.")
        return 1

    faithfulness = _coerce_float_series(df, "faithfulness")
    answer_relevancy = _coerce_float_series(df, "answer_relevancy")
    context_precision = _coerce_float_series(df, "context_precision")
    context_recall = _coerce_float_series(df, "context_recall")

    views: list[RowView] = []
    for i in range(len(df)):
        q = str(df.loc[i, "question"]) if pd.notna(df.loc[i, "question"]) else ""
        q = q.strip()
        evo = str(df.loc[i, "evolution_type"]).strip() if "evolution_type" in df.columns and pd.notna(df.loc[i, "evolution_type"]) else ""
        if not evo and q in evo_map:
            evo = evo_map[q]

        row = {
            "faithfulness": None if pd.isna(faithfulness.iloc[i]) else float(faithfulness.iloc[i]),
            "answer_relevancy": None if pd.isna(answer_relevancy.iloc[i]) else float(answer_relevancy.iloc[i]),
            "context_precision": None if pd.isna(context_precision.iloc[i]) else float(context_precision.iloc[i]),
            "context_recall": None if pd.isna(context_recall.iloc[i]) else float(context_recall.iloc[i]),
        }
        avg = _compute_avg(row)
        cluster = _assign_cluster(
            evolution_type=evo,
            faithfulness=row["faithfulness"],
            answer_relevancy=row["answer_relevancy"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
        )
        views.append(
            RowView(
                question=q,
                evolution_type=evo,
                faithfulness=row["faithfulness"],
                answer_relevancy=row["answer_relevancy"],
                context_precision=row["context_precision"],
                context_recall=row["context_recall"],
                avg=avg,
                cluster=cluster,
            )
        )

    # Sort by avg ascending; treat missing avg as 0.0 to surface unknown/missing metrics.
    def sort_key(v: RowView) -> float:
        if v.avg is None or pd.isna(v.avg):
            return 0.0
        return float(v.avg)

    bottom = sorted(views, key=sort_key)[:10]

    # Overall aggregate means (where possible)
    summary: dict[str, float] = {}
    for m, s in [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]:
        if len(s) == 0:
            continue
        mean = pd.to_numeric(s, errors="coerce").mean()
        if pd.notna(mean):
            summary[m] = float(mean)

    md_parts: list[str] = []
    md_parts.append("# Failure Analysis (Phase A)\n\n")
    if summary:
        md_parts.append("## Aggregate Metrics (Mean)\n\n")
        md_parts.append(
            "\n".join([f"- {k}: {v:.3f}" for k, v in summary.items()]) + "\n\n"
        )

    md_parts.append("## Bottom 10 Failures (by Avg)\n\n")
    md_parts.append(_make_markdown_table(bottom) + "\n")

    md_parts.append("## Clusters & Recommendations\n\n")
    md_parts.append(_cluster_sections(bottom))

    out_md.write_text("".join(md_parts), encoding="utf-8")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

