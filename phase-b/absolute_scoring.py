from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


JUDGE_MODEL = "gpt-4o-mini"
DIMENSIONS = ["accuracy", "relevance", "conciseness", "helpfulness"]


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment (or copy `.env.example` to `.env`) "
            "before running absolute scoring."
        )


def _load_rows(path: Path, n: int) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    if "question" not in rows[0] or "answer_a" not in rows[0]:
        raise RuntimeError("pairwise_results.csv must contain columns: question, answer_a")
    return rows[:n]


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> str:
    s = _strip_code_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge did not return a JSON object.")
    return s[start : end + 1]


def _parse_scores(text: str) -> dict[str, Any]:
    obj = json.loads(_extract_json_object(text))
    if not isinstance(obj, dict):
        raise ValueError("Judge JSON is not an object.")
    return obj


def _coerce_score(x: Any) -> int:
    try:
        v = int(x)
    except Exception as e:
        raise ValueError(f"Invalid score {x!r}: {e}") from e
    if v < 1 or v > 5:
        raise ValueError(f"Score out of range (1-5): {v}")
    return v


def _judge_prompt(question: str, answer: str) -> str:
    return (
        "You are grading a single answer to a question.\n"
        "Score the answer on four independent dimensions (1-5 integers):\n"
        "- accuracy (is it factually correct and not hallucinated?)\n"
        "- relevance (does it answer the question asked?)\n"
        "- conciseness (is it brief without losing key info?)\n"
        "- helpfulness (is it actionable, clear, and useful?)\n\n"
        "Return JSON only, with no markdown and no extra keys:\n"
        '{"accuracy":1-5,"relevance":1-5,"conciseness":1-5,"helpfulness":1-5,"reason":"..."}\n\n'
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n"
    )


def _score_one(llm: ChatOpenAI, question: str, answer: str) -> tuple[dict[str, int], str]:
    raw = llm.invoke(_judge_prompt(question, answer)).content
    data = _parse_scores(str(raw))
    scores = {d: _coerce_score(data.get(d)) for d in DIMENSIONS}
    reason = str(data.get("reason", "")).strip()
    return scores, reason


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Absolute scoring (LLM-as-judge) for answer_a.")
    parser.add_argument("--n", type=int, default=30, help="Number of rows to score (default 30).")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="phase-b/pairwise_results.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="phase-b/absolute_scores.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    try:
        _require_openai_api_key()
        rows = _load_rows(Path(args.in_path), n=int(args.n))
        llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

        out_rows: list[dict[str, Any]] = []
        for r in rows:
            q = (r.get("question") or "").strip()
            a = (r.get("answer_a") or "").strip()
            if not q or not a:
                continue

            scores, reason = _score_one(llm, q, a)
            overall = sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS)

            out_rows.append(
                {
                    "question": q,
                    "answer": a,
                    **scores,
                    "overall": f"{overall:.2f}",
                    "reason": reason,
                }
            )

        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["question", "answer", *DIMENSIONS, "overall", "reason"]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        print(f"Wrote: {out_path} ({len(out_rows)} rows)")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

