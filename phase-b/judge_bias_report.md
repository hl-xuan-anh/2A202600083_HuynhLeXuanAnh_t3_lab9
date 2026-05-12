# Judge Bias Report (Phase B)

This report quantifies common LLM-as-judge biases using `phase-b/pairwise_results.csv`.

## Summary Numbers

| Metric | Value |
|---|---:|
| Rows | 30 |
| Position bias: run1 winner = A | 6.7% |
| Position bias: run1 winner = B | 13.3% |
| Position bias: run1 winner = tie | 80.0% |
| Swap agreement rate (run1 == run2) | 96.7% |
| Length bias: shorter answer wins | 0.0% |
| Length bias: longer answer wins | 16.7% |
| Length bias: tie | 83.3% |

## Chart

See `phase-b/judge_bias_chart.png` for winner distributions vs answer-length differences.

## Mitigation Strategy

- **Swap-and-average:** Already reduces position/order bias by judging both (A,B) and (B,A) and requiring agreement.
- **Stricter rubric:** Constrain the judge to explicit criteria and require citations to retrieved context when possible.
- **Length-normalized prompt:** Instruct the judge to ignore verbosity and score content quality; optionally truncate or summarize answers to similar lengths before judging.
