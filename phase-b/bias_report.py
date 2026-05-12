from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class BiasStats:
    n: int
    run1_a_rate: float
    run1_b_rate: float
    run1_tie_rate: float
    swap_agreement_rate: float
    length_bias_shorter_win_rate: float
    length_bias_longer_win_rate: float
    length_bias_tie_rate: float


def _safe_rate(x: int, n: int) -> float:
    return 0.0 if n <= 0 else x / n


def _winner_norm(x: object) -> str:
    s = str(x or "").strip().lower()
    if s == "a":
        return "A"
    if s == "b":
        return "B"
    if s in {"tie", "draw", "equal"}:
        return "tie"
    return "tie"


def _compute_stats(df: pd.DataFrame) -> BiasStats:
    n = len(df)
    run1 = df.get("run1_winner", pd.Series(["tie"] * n)).map(_winner_norm)
    run2 = df.get("run2_winner", pd.Series(["tie"] * n)).map(_winner_norm)
    after = df.get("winner_after_swap", pd.Series(["tie"] * n)).map(_winner_norm)

    run1_a = int((run1 == "A").sum())
    run1_b = int((run1 == "B").sum())
    run1_t = int((run1 == "tie").sum())

    # Swap agreement: run1_winner equals run2_winner (already flipped in our CSV)
    agree = int((run1 == run2).sum())

    # Length bias: compare answer lengths vs winner_after_swap.
    a_len = df.get("answer_a", pd.Series([""] * n)).fillna("").map(lambda s: len(str(s)))
    b_len = df.get("answer_b", pd.Series([""] * n)).fillna("").map(lambda s: len(str(s)))

    shorter_wins = 0
    longer_wins = 0
    ties = 0
    for i in range(n):
        w = after.iloc[i]
        la = int(a_len.iloc[i])
        lb = int(b_len.iloc[i])
        if w == "tie":
            ties += 1
            continue
        if la == lb:
            # no length signal; count neither
            continue
        if (w == "A" and la < lb) or (w == "B" and lb < la):
            shorter_wins += 1
        elif (w == "A" and la > lb) or (w == "B" and lb > la):
            longer_wins += 1

    decisive = shorter_wins + longer_wins
    # Report rates over all rows for simplicity; also works when many ties.
    return BiasStats(
        n=n,
        run1_a_rate=_safe_rate(run1_a, n),
        run1_b_rate=_safe_rate(run1_b, n),
        run1_tie_rate=_safe_rate(run1_t, n),
        swap_agreement_rate=_safe_rate(agree, n),
        length_bias_shorter_win_rate=_safe_rate(shorter_wins, n),
        length_bias_longer_win_rate=_safe_rate(longer_wins, n),
        length_bias_tie_rate=_safe_rate(ties, n),
    )


def _make_chart(df: pd.DataFrame, out_path: Path) -> None:
    n = len(df)
    if n == 0:
        return

    after = df.get("winner_after_swap", pd.Series(["tie"] * n)).map(_winner_norm)
    a_len = df.get("answer_a", pd.Series([""] * n)).fillna("").map(lambda s: len(str(s)))
    b_len = df.get("answer_b", pd.Series([""] * n)).fillna("").map(lambda s: len(str(s)))
    diff = (a_len - b_len).astype(int)  # positive => A longer

    # Bucket length differences for a simple, readable plot
    buckets = []
    labels = []
    for lo, hi, label in [(-10_000, -400, "<= -400"), (-399, -100, "-399..-100"), (-99, 99, "-99..99"), (100, 399, "100..399"), (400, 10_000, ">= 400")]:
        mask = (diff >= lo) & (diff <= hi)
        sub = after[mask]
        buckets.append([int((sub == "A").sum()), int((sub == "B").sum()), int((sub == "tie").sum())])
        labels.append(label)

    a_counts = [b[0] for b in buckets]
    b_counts = [b[1] for b in buckets]
    t_counts = [b[2] for b in buckets]

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, a_counts, label="Winner=A")
    plt.bar(x, b_counts, bottom=a_counts, label="Winner=B")
    bottoms = [a_counts[i] + b_counts[i] for i in range(len(labels))]
    plt.bar(x, t_counts, bottom=bottoms, label="Winner=tie")

    plt.xticks(list(x), labels)
    plt.ylabel("Count")
    plt.xlabel("Length diff (len(answer_a) - len(answer_b))")
    plt.title("Winner after swap vs answer length difference")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _write_report(stats: BiasStats, out_md: Path) -> None:
    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    table = "\n".join(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| Rows | {stats.n} |",
            f"| Position bias: run1 winner = A | {pct(stats.run1_a_rate)} |",
            f"| Position bias: run1 winner = B | {pct(stats.run1_b_rate)} |",
            f"| Position bias: run1 winner = tie | {pct(stats.run1_tie_rate)} |",
            f"| Swap agreement rate (run1 == run2) | {pct(stats.swap_agreement_rate)} |",
            f"| Length bias: shorter answer wins | {pct(stats.length_bias_shorter_win_rate)} |",
            f"| Length bias: longer answer wins | {pct(stats.length_bias_longer_win_rate)} |",
            f"| Length bias: tie | {pct(stats.length_bias_tie_rate)} |",
        ]
    )

    out_md.write_text(
        "\n".join(
            [
                "# Judge Bias Report (Phase B)",
                "",
                "This report quantifies common LLM-as-judge biases using `phase-b/pairwise_results.csv`.",
                "",
                "## Summary Numbers",
                "",
                table,
                "",
                "## Chart",
                "",
                "See `phase-b/judge_bias_chart.png` for winner distributions vs answer-length differences.",
                "",
                "## Mitigation Strategy",
                "",
                "- **Swap-and-average:** Already reduces position/order bias by judging both (A,B) and (B,A) and requiring agreement.",
                "- **Stricter rubric:** Constrain the judge to explicit criteria and require citations to retrieved context when possible.",
                "- **Length-normalized prompt:** Instruct the judge to ignore verbosity and score content quality; optionally truncate or summarize answers to similar lengths before judging.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    in_path = Path("phase-b") / "pairwise_results.csv"
    out_md = Path("phase-b") / "judge_bias_report.md"
    out_png = Path("phase-b") / "judge_bias_chart.png"

    if not in_path.exists():
        print(f"ERROR: Missing file: {in_path}")
        return 1

    df = pd.read_csv(in_path)
    if len(df) == 0:
        print("ERROR: pairwise_results.csv is empty.")
        return 1

    stats = _compute_stats(df)
    _make_chart(df, out_png)
    _write_report(stats, out_md)

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_png}")
    print(
        "Concise summary: "
        f"run1(A)={stats.run1_a_rate:.3f}  run1(B)={stats.run1_b_rate:.3f}  "
        f"swap_agree={stats.swap_agreement_rate:.3f}  shorter_win={stats.length_bias_shorter_win_rate:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

