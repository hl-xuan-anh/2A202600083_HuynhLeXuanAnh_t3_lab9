from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class AlignedLabels:
    n: int
    human: list[str]
    judge: list[str]
    mode: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _norm_label(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"a", "answer_a", "left"}:
        return "A"
    if s in {"b", "answer_b", "right"}:
        return "B"
    if s in {"tie", "draw", "equal"}:
        return "tie"
    # Unknown / missing -> tie (neutral) to keep script useful
    return "tie"


def _align(human_rows: list[dict[str, str]], judge_rows: list[dict[str, str]]) -> AlignedLabels:
    # Prefer question_id matching if available in BOTH.
    human_has_qid = human_rows and ("question_id" in human_rows[0] or "id" in human_rows[0])
    judge_has_qid = judge_rows and ("question_id" in judge_rows[0] or "id" in judge_rows[0])

    if human_has_qid and judge_has_qid:
        def get_id(r: dict[str, str]) -> str:
            return (r.get("question_id") or r.get("id") or "").strip()

        human_map = {get_id(r): r for r in human_rows if get_id(r)}
        judge_map = {get_id(r): r for r in judge_rows if get_id(r)}
        common = [k for k in human_map.keys() if k in judge_map]
        common = common[: max(10, len(common))]
        if common:
            human = [_norm_label(human_map[k].get("label", "")) for k in common]
            judge = [_norm_label(judge_map[k].get("winner_after_swap", judge_map[k].get("run1_winner", ""))) for k in common]
            return AlignedLabels(n=len(common), human=human, judge=judge, mode="question_id")

    # Fallback: first 10 rows (as requested).
    n = min(10, len(human_rows), len(judge_rows))
    if n == 0:
        return AlignedLabels(n=0, human=[], judge=[], mode="none")

    human = [_norm_label(human_rows[i].get("label", "")) for i in range(n)]
    judge = [
        _norm_label(judge_rows[i].get("winner_after_swap", judge_rows[i].get("run1_winner", "")))
        for i in range(n)
    ]
    return AlignedLabels(n=n, human=human, judge=judge, mode="first_10")


def _interpret(kappa: float) -> str:
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


def main() -> int:
    human_path = Path("phase-b") / "human_labels.csv"
    judge_path = Path("phase-b") / "pairwise_results.csv"
    out_md = Path("phase-b") / "kappa_summary.md"

    try:
        human_rows = _read_csv(human_path)
        judge_rows = _read_csv(judge_path)
        aligned = _align(human_rows, judge_rows)
        if aligned.n == 0:
            raise RuntimeError("No overlapping rows to compare (check CSV contents).")

        kappa = float(cohen_kappa_score(aligned.human, aligned.judge))
        interpretation = _interpret(kappa)

        print(f"Cohen's kappa (n={aligned.n}, mode={aligned.mode}): {kappa:.3f}  ->  {interpretation}")

        out_md.write_text(
            "\n".join(
                [
                    "# Cohen's Kappa Summary",
                    "",
                    f"- Compared rows: {aligned.n}",
                    f"- Alignment mode: `{aligned.mode}`",
                    f"- Kappa: `{kappa:.3f}`",
                    f"- Interpretation: **{interpretation}**",
                    "",
                    "## Legend",
                    "",
                    "- <0: worse than chance",
                    "- <0.2: slight",
                    "- <0.4: fair",
                    "- <0.6: moderate",
                    "- <0.8: substantial",
                    "- else: almost perfect",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {out_md}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

