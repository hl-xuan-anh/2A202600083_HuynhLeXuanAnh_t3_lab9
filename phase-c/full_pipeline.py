from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running as `python phase-c/full_pipeline.py` with repo-root imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_c_imports import InputGuard, OutputGuardAPI, TopicGuard  # noqa: E402
from rag_pipeline import my_rag_pipeline_async  # noqa: E402


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


async def _audit_log_fire_and_forget(event: dict[str, Any]) -> None:
    """
    Fire-and-forget async audit log: appends one JSONL line to phase-c/audit_log.jsonl.
    """

    async def _write() -> None:
        path = Path("phase-c") / "audit_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        def _append() -> None:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_append)

    # Detached task; do not await.
    asyncio.create_task(_write())


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    timings_ms: dict[str, float]
    blocked: bool
    reason: str


async def guarded_pipeline(user_input: str) -> tuple[str, dict[str, float]]:
    """
    L1: PII sanitize + topic check
    L2: RAG pipeline
    L3: Output guard check
    L4: audit log (fire-and-forget)
    Returns: (answer, timings_ms)
    """
    t0 = _now_ms()

    if not user_input.strip():
        return "Empty input.", {"l1_ms": 0.0, "l2_ms": 0.0, "l3_ms": 0.0, "total_ms": 0.0}

    # L1
    l1s = _now_ms()
    input_guard = InputGuard()
    sanitized, sanitize_ms = await input_guard.sanitize_async(user_input)

    allowed_topics = [
        "RAG pipelines",
        "evaluation (RAGAS metrics)",
        "guardrails (PII, prompt injection, topic scoping)",
        "LangChain-based document retrieval and generation",
    ]
    topic_guard = TopicGuard(allowed_topics=allowed_topics)
    on_topic, topic_reason = await topic_guard.check_async(sanitized)
    l1e = _now_ms()
    l1_ms = float(l1e - l1s)

    if not on_topic:
        await _audit_log_fire_and_forget(
            {
                "stage": "L1",
                "blocked": True,
                "reason": topic_reason,
                "input": user_input,
                "sanitized": sanitized,
                "timings_ms": {"l1_ms": l1_ms},
            }
        )
        total_ms = float(_now_ms() - t0)
        return topic_reason, {"l1_ms": l1_ms, "l2_ms": 0.0, "l3_ms": 0.0, "total_ms": total_ms}

    # L2
    l2s = _now_ms()
    answer, _contexts = await my_rag_pipeline_async(sanitized)
    l2e = _now_ms()
    l2_ms = float(l2e - l2s)

    # L3
    l3s = _now_ms()
    output_guard = OutputGuardAPI()
    is_safe, raw, _og_ms = await output_guard.check_async(user_input, answer)
    l3e = _now_ms()
    l3_ms = float(l3e - l3s)

    final_answer = answer if is_safe else "Blocked by output safety policy."

    total_ms = float(_now_ms() - t0)
    timings = {"l1_ms": l1_ms, "l2_ms": l2_ms, "l3_ms": l3_ms, "total_ms": total_ms}

    await _audit_log_fire_and_forget(
        {
            "stage": "L4",
            "blocked": (not is_safe),
            "reason": raw,
            "input": user_input,
            "sanitized": sanitized,
            "answer": final_answer,
            "timings_ms": timings,
        }
    )

    return final_answer, timings


async def baseline_pipeline(user_input: str) -> tuple[str, dict[str, float]]:
    """
    Baseline without guardrails: just calls the async RAG pipeline.
    """
    t0 = _now_ms()
    l2s = _now_ms()
    answer, _contexts = await my_rag_pipeline_async(user_input)
    l2e = _now_ms()
    total_ms = float(_now_ms() - t0)
    return answer, {"l2_ms": float(l2e - l2s), "total_ms": total_ms}


def _write_latency_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["request_id", "mode", "l1_ms", "l2_ms", "l3_ms", "total_ms"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _print_percentiles(name: str, values: list[float]) -> None:
    print(
        f"{name}: P50={_pct(values,0.50):.2f}ms  P95={_pct(values,0.95):.2f}ms  P99={_pct(values,0.99):.2f}ms"
    )


async def benchmark(n: int = 100) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Missing OPENAI_API_KEY. Benchmark requires TopicGuard + RAG pipeline.")
        return 1

    queries = [
        "What is RAG?",
        "What is faithfulness in RAGAS?",
        "How do I chunk markdown documents for a vector store?",
        "What is context precision vs context recall?",
        "How can I reduce hallucinations in a RAG system?",
        "Explain swap-and-average in LLM-as-judge.",
        "What guardrails help prevent PII leakage?",
        "How do I persist and load a local Chroma vector store?",
        "What are common retrieval failure modes?",
        "How do I set thresholds for evaluation metrics?",
    ]

    guarded_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []

    guarded_l1: list[float] = []
    guarded_l2: list[float] = []
    guarded_l3: list[float] = []
    guarded_total: list[float] = []
    base_l2: list[float] = []
    base_total: list[float] = []

    for i in range(n):
        q = queries[i % len(queries)]

        _ans_g, t_g = await guarded_pipeline(q)
        guarded_rows.append(
            {
                "request_id": i,
                "mode": "guarded",
                "l1_ms": f"{t_g.get('l1_ms',0.0):.2f}",
                "l2_ms": f"{t_g.get('l2_ms',0.0):.2f}",
                "l3_ms": f"{t_g.get('l3_ms',0.0):.2f}",
                "total_ms": f"{t_g.get('total_ms',0.0):.2f}",
            }
        )
        guarded_l1.append(float(t_g.get("l1_ms", 0.0)))
        guarded_l2.append(float(t_g.get("l2_ms", 0.0)))
        guarded_l3.append(float(t_g.get("l3_ms", 0.0)))
        guarded_total.append(float(t_g.get("total_ms", 0.0)))

        _ans_b, t_b = await baseline_pipeline(q)
        base_rows.append(
            {
                "request_id": i,
                "mode": "baseline",
                "l1_ms": "0.00",
                "l2_ms": f"{t_b.get('l2_ms',0.0):.2f}",
                "l3_ms": "0.00",
                "total_ms": f"{t_b.get('total_ms',0.0):.2f}",
            }
        )
        base_l2.append(float(t_b.get("l2_ms", 0.0)))
        base_total.append(float(t_b.get("total_ms", 0.0)))

    out_csv = Path("phase-c") / "latency_benchmark.csv"
    _write_latency_csv(guarded_rows + base_rows, out_csv)

    print(f"Wrote: {out_csv}")
    print("\nGuarded percentiles:")
    _print_percentiles("L1", guarded_l1)
    _print_percentiles("L2", guarded_l2)
    _print_percentiles("L3", guarded_l3)
    _print_percentiles("Total", guarded_total)

    print("\nBaseline percentiles:")
    _print_percentiles("L2", base_l2)
    _print_percentiles("Total", base_total)

    overhead_p50 = _pct(guarded_total, 0.50) - _pct(base_total, 0.50)
    overhead_p95 = _pct(guarded_total, 0.95) - _pct(base_total, 0.95)
    overhead_p99 = _pct(guarded_total, 0.99) - _pct(base_total, 0.99)
    print(f"\nOverhead (guarded - baseline): P50={overhead_p50:.2f}ms  P95={overhead_p95:.2f}ms  P99={overhead_p99:.2f}ms")
    return 0


def main() -> int:
    load_dotenv(override=False)
    try:
        return asyncio.run(benchmark(n=100))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
