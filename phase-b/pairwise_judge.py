from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


JUDGE_MODEL = "gpt-4o-mini"

# Allow running as `python phase-b/pairwise_judge.py` while importing repo-root modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from your_rag_module import my_rag_pipeline  # noqa: E402


def _coerce_pipeline_output(out: Any) -> str:
    """
    Accepts either:
      - (answer, contexts)
      - {"answer": ..., "contexts": ...}
    Returns answer as string.
    """
    if isinstance(out, dict):
        return str(out.get("answer", "")).strip()
    if isinstance(out, tuple) and len(out) == 2:
        return str(out[0]).strip()
    raise RuntimeError("my_rag_pipeline returned an unsupported type; expected (answer, contexts) or dict.")


def get_answer_v1(question: str) -> str:
    # TODO: Replace with your true v1 system (prompting/retrieval settings/etc.)
    return _coerce_pipeline_output(my_rag_pipeline(question))


def get_answer_v2(question: str) -> str:
    # TODO: Replace with your true v2 system (prompting/retrieval settings/etc.)
    return _coerce_pipeline_output(my_rag_pipeline(question))


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment (or copy `.env.example` to `.env`) "
            "before running the judge."
        )


def _load_questions(testset_path: Path, n: int) -> list[str]:
    if not testset_path.exists():
        raise FileNotFoundError(f"Missing testset file: {testset_path}")
    with testset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise RuntimeError(f"Empty testset: {testset_path}")

    # Prefer "question" column, fallback to "query" if needed.
    qs: list[str] = []
    for r in rows:
        q = (r.get("question") or r.get("query") or "").strip()
        if q:
            qs.append(q)
        if len(qs) >= n:
            break
    if len(qs) < min(30, n):
        raise RuntimeError(f"Need at least 30 questions; found {len(qs)} in {testset_path}")
    return qs


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove ```json ... ``` or ``` ... ```
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> str:
    """
    Extract the first {...} JSON object from a string.
    """
    s = _strip_code_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge did not return a JSON object.")
    return s[start : end + 1]


def _parse_judge_json(text: str) -> tuple[str, str]:
    """
    Returns (winner, reason) where winner in {"A","B","tie"}.
    Robust to markdown fences and extra text around JSON.
    """
    obj_text = _extract_json_object(text)
    data = json.loads(obj_text)
    if not isinstance(data, dict):
        raise ValueError("Judge JSON is not an object.")
    winner = str(data.get("winner", "")).strip()
    reason = str(data.get("reason", "")).strip()

    winner_norm = winner.lower()
    if winner_norm in {"a", "answer_a"}:
        return "A", reason
    if winner_norm in {"b", "answer_b"}:
        return "B", reason
    if winner_norm in {"tie", "draw", "equal"}:
        return "tie", reason
    raise ValueError(f"Invalid winner value: {winner!r} (expected 'A', 'B', or 'tie').")


def _judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return (
        "You are an impartial judge comparing two answers to the same question.\n"
        "Decide which answer is better overall given correctness, completeness, and grounding.\n"
        "If they are equivalent, choose tie.\n\n"
        "Return JSON only, with no markdown:\n"
        '{"winner":"A"|"B"|"tie","reason":"..."}\n\n'
        f"Question:\n{question}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n"
    )


@dataclass(frozen=True)
class JudgeRun:
    winner: str  # A, B, tie
    reason: str
    raw: str


def _run_judge(llm: ChatOpenAI, question: str, answer_a: str, answer_b: str) -> JudgeRun:
    prompt = _judge_prompt(question, answer_a, answer_b)
    raw = llm.invoke(prompt).content
    winner, reason = _parse_judge_json(str(raw))
    return JudgeRun(winner=winner, reason=reason, raw=str(raw))


def _flip_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


def _agreed_winner(run1: str, run2_flipped: str) -> str:
    if run1 == run2_flipped:
        return run1
    return "tie"


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Pairwise LLM-as-judge with swap-and-average.")
    parser.add_argument("--n", type=int, default=30, help="Number of questions to run (min 30).")
    parser.add_argument(
        "--testset",
        type=str,
        default="phase-a/testset_v1.csv",
        help="Path to the Phase A testset CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="phase-b/pairwise_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    try:
        _require_openai_api_key()
        n = max(30, int(args.n))
        questions = _load_questions(Path(args.testset), n=n)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

        rows: list[dict[str, Any]] = []
        for q in questions:
            answer_v1 = get_answer_v1(q)
            answer_v2 = get_answer_v2(q)

            run1 = _run_judge(llm, q, answer_v1, answer_v2)
            run2 = _run_judge(llm, q, answer_v2, answer_v1)
            run2_winner_flipped = _flip_winner(run2.winner)
            agreed = _agreed_winner(run1.winner, run2_winner_flipped)

            rows.append(
                {
                    "question": q,
                    "answer_a": answer_v1,
                    "answer_b": answer_v2,
                    "run1_winner": run1.winner,
                    "run2_winner": run2_winner_flipped,
                    "winner_after_swap": agreed,
                    "run1_reason": run1.reason,
                    "run2_reason": run2.reason,
                }
            )

        # Write CSV
        fieldnames = [
            "question",
            "answer_a",
            "answer_b",
            "run1_winner",
            "run2_winner",
            "winner_after_swap",
            "run1_reason",
            "run2_reason",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        print(f"Wrote: {out_path} ({len(rows)} rows)")
        return 0
    except NotImplementedError as e:
        print(f"ERROR: {e}")
        print("Implement `get_answer_v1` and `get_answer_v2` in phase-b/pairwise_judge.py.")
        return 2
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
