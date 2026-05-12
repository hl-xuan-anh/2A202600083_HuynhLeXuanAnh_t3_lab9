from __future__ import annotations

import asyncio
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VN_CCCD_RE = re.compile(r"\b\d{12}\b")
VN_PHONE_RE = re.compile(r"\b(?:\+84|0)\d{9,10}\b")
VN_TAX_RE = re.compile(r"\b\d{10}(?:-\d{3})?\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class SanitizeResult:
    output: str
    pii_found: bool
    latency_ms: float


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(round(0.95 * (len(xs) - 1)))
    return float(xs[max(0, min(k, len(xs) - 1))])


class InputGuard:
    """
    Input guard that scrubs obvious Vietnamese PII by regex first,
    then applies Presidio anonymization for additional detections.
    """

    def __init__(self) -> None:
        self._presidio = None

    def _get_presidio(self):
        if self._presidio is not None:
            return self._presidio

        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
            from presidio_anonymizer import AnonymizerEngine  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"Presidio is not available ({e}). Install `presidio-analyzer` and `presidio-anonymizer`."
            ) from e

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        self._presidio = (analyzer, anonymizer)
        return self._presidio

    def scrub_vn(self, text: str) -> tuple[str, bool]:
        if not text:
            return text, False

        pii = False

        def sub(pattern: re.Pattern[str], repl: str, s: str) -> str:
            nonlocal pii
            if pattern.search(s):
                pii = True
            return pattern.sub(repl, s)

        out = text
        out = sub(EMAIL_RE, "[REDACTED_EMAIL]", out)
        out = sub(VN_PHONE_RE, "[REDACTED_PHONE_VN]", out)
        out = sub(VN_CCCD_RE, "[REDACTED_CCCD]", out)
        out = sub(VN_TAX_RE, "[REDACTED_TAX_CODE]", out)
        return out, pii

    def scrub_ner(self, text: str) -> tuple[str, bool]:
        if not text:
            return text, False

        analyzer, anonymizer = self._get_presidio()

        # Presidio language coverage varies; use English pipeline which includes useful recognizers.
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE", "US_SSN"],
        )
        pii = bool(results)
        if not pii:
            return text, False

        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, True

    def sanitize(self, text: str) -> tuple[str, float]:
        start = _now_ms()
        out, _pii1 = self.scrub_vn(text)
        out, _pii2 = self.scrub_ner(out)
        end = _now_ms()
        return out, float(end - start)

    async def sanitize_async(self, text: str) -> tuple[str, float]:
        # Presidio is CPU-bound; run in a thread for async usage.
        return await asyncio.to_thread(self.sanitize, text)


def _pii_found_by_regex(text: str) -> bool:
    if not text:
        return False
    return bool(
        EMAIL_RE.search(text)
        or VN_PHONE_RE.search(text)
        or VN_CCCD_RE.search(text)
        or VN_TAX_RE.search(text)
    )


def _write_csv(rows: Iterable[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input", "output", "pii_found", "latency_ms"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    guard = InputGuard()

    long_text = (
        "This is a long prompt. " * 200
        + "Contact: test.user@example.com and phone 0912345678 and CCCD 012345678901."
    )

    tests = [
        "",  # empty
        "Hello, how are you?",  # English, no PII
        "My email is alice@example.com, please reply.",  # email
        "Số điện thoại của tôi là 0912345678",  # VN phone
        "Liên hệ +849123456789 để biết thêm chi tiết.",  # +84 phone
        "CCCD của tôi: 012345678901",  # cccd
        "Mã số thuế: 1234567890-001",  # tax
        "Mixed PII: email bob@company.vn phone 0387654321 CCCD 123456789012",  # mixed
        long_text,  # long string with PII
        "No PII but looks like code: ```json {\"a\":1} ```",  # markdown fences
    ]

    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    found = 0

    for t in tests:
        start = _now_ms()
        out1, pii1 = guard.scrub_vn(t)
        out2, pii2 = guard.scrub_ner(out1)
        end = _now_ms()
        output = out2
        latency_ms = float(end - start)
        pii_found = bool(pii1 or pii2)
        if pii_found:
            found += 1
        latencies.append(latency_ms)
        rows.append(
            {
                "input": t,
                "output": output,
                "pii_found": bool(pii_found),
                "latency_ms": f"{latency_ms:.2f}",
            }
        )

    out_csv = Path("phase-c") / "pii_test_results.csv"
    _write_csv(rows, out_csv)

    detection_rate = found / len(tests) if tests else 0.0
    p95 = _p95(latencies)

    print(f"Wrote: {out_csv}")
    print(f"Detection rate: {detection_rate:.2f} ({found}/{len(tests)})")
    print(f"P95 latency (ms): {p95:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
