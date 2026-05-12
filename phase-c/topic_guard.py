from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


JUDGE_MODEL = "gpt-4o-mini"


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment (or copy `.env.example` to `.env`) "
            "before running TopicGuard."
        )


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
        raise ValueError("Model did not return a JSON object.")
    return s[start : end + 1]


def _parse_guard_json(text: str) -> tuple[bool, str]:
    data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("Guard JSON is not an object.")
    allowed = bool(data.get("allowed", False))
    reason = str(data.get("reason", "")).strip()
    return allowed, reason


@dataclass(frozen=True)
class TopicGuardResult:
    allowed: bool
    reason: str


class TopicGuard:
    """
    Zero-shot topic classifier for scoping an assistant to specific topics.

    - allowed_topics: list of allowed scope topics (configurable)
    - check(text) -> (allowed: bool, reason: str)
    """

    def __init__(self, allowed_topics: list[str]) -> None:
        self.allowed_topics = [t.strip() for t in allowed_topics if t.strip()]
        if not self.allowed_topics:
            raise ValueError("allowed_topics must be non-empty.")
        self._llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    def _fallback_message(self) -> str:
        topics = ", ".join(self.allowed_topics)
        return f"I can only answer questions about {topics}. Please rephrase within scope."

    def _prompt(self, text: str) -> str:
        topics = "\n".join([f"- {t}" for t in self.allowed_topics])
        return (
            "You are a strict topic guard for an assistant.\n"
            "Decide if the user's input is within the allowed topics.\n"
            "Be conservative: if unsure, mark it as not allowed.\n\n"
            "Allowed topics:\n"
            f"{topics}\n\n"
            "Return JSON only (no markdown), exactly with keys:\n"
            '{"allowed": true|false, "reason": "short reason"}\n\n'
            f"User input:\n{text}"
        )

    def check(self, text: str) -> tuple[bool, str]:
        if not text.strip():
            return False, self._fallback_message()

        raw = self._llm.invoke([HumanMessage(content=self._prompt(text))]).content
        try:
            allowed, reason = _parse_guard_json(str(raw))
        except Exception:
            # If parsing fails, fail closed with a helpful message.
            return False, self._fallback_message()

        if allowed:
            return True, reason or "Allowed."
        return False, self._fallback_message()

    async def check_async(self, text: str) -> tuple[bool, str]:
        if not text.strip():
            return False, self._fallback_message()

        msg = HumanMessage(content=self._prompt(text))
        if hasattr(self._llm, "ainvoke"):
            raw = (await self._llm.ainvoke([msg])).content  # type: ignore[call-arg]
        else:
            raw = await asyncio.to_thread(self._llm.invoke, [msg])
            raw = getattr(raw, "content", raw)

        try:
            allowed, reason = _parse_guard_json(str(raw))
        except Exception:
            return False, self._fallback_message()

        if allowed:
            return True, reason or "Allowed."
        return False, self._fallback_message()


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input", "expected_on_topic", "allowed", "reason"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    load_dotenv(override=False)
    try:
        _require_openai_api_key()
        allowed_topics = [
            "RAG pipelines",
            "evaluation (RAGAS metrics)",
            "guardrails (PII, prompt injection, topic scoping)",
            "LangChain-based document retrieval and generation",
        ]
        guard = TopicGuard(allowed_topics=allowed_topics)

        on_topic = [
            "How do I build an index for markdown docs in a RAG pipeline?",
            "What is context precision in RAGAS?",
            "How can I reduce hallucinations in RAG?",
            "Explain retrieval augmented generation briefly.",
            "What guardrails help prevent PII leakage?",
            "How do I chunk documents for a vector store?",
            "How can I evaluate faithfulness for RAG answers?",
            "What is swap-and-average in LLM-as-judge evaluations?",
            "How can I detect prompt injection attempts in user input?",
            "How do I persist a Chroma vectorstore locally?",
        ]
        off_topic = [
            "Write me a love poem in Vietnamese.",
            "What is the best stock to buy today?",
            "Solve this calculus integral step by step.",
            "Give me a recipe for pho.",
            "Who will win the next football match?",
            "Explain the French Revolution.",
            "Generate a Minecraft mod idea list.",
            "Help me debug my printer driver.",
            "Translate this sentence to Japanese.",
            "Plan a 7-day trip to Japan.",
        ]

        tests: list[tuple[str, bool]] = [(t, True) for t in on_topic] + [(t, False) for t in off_topic]

        rows: list[dict[str, object]] = []
        correct = 0
        refused = 0

        for text, expected_on_topic in tests:
            allowed, reason = guard.check(text)
            pred_on_topic = bool(allowed)
            if pred_on_topic == expected_on_topic:
                correct += 1
            if not allowed:
                refused += 1
            rows.append(
                {
                    "input": text,
                    "expected_on_topic": expected_on_topic,
                    "allowed": allowed,
                    "reason": reason,
                }
            )

        out_path = Path("phase-c") / "topic_guard_results.csv"
        _write_csv(rows, out_path)

        accuracy = correct / len(tests) if tests else 0.0
        refuse_rate = refused / len(tests) if tests else 0.0

        print(f"Wrote: {out_path}")
        print(f"Accuracy: {accuracy:.2f} ({correct}/{len(tests)})")
        print(f"Refuse rate: {refuse_rate:.2f} ({refused}/{len(tests)})")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

