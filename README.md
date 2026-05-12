# RAG Evaluation & Guardrails Lab

## Overview (200–300 words)

This repository is a compact, production-oriented lab that builds and evaluates a Retrieval-Augmented Generation (RAG) system and adds practical guardrails. The workflow is organized into phases: **Phase A** generates a synthetic evaluation set and runs **RAGAS** metrics (faithfulness, answer relevancy, context precision/recall) with threshold gating suitable for CI. **Phase B** applies LLM-as-judge evaluation patterns: pairwise comparisons with swap-and-average, absolute scoring across multiple dimensions, inter-annotator agreement (Cohen’s kappa), and bias reporting (position and length bias). **Phase C** implements guardrails and operational checks: Vietnamese PII redaction and Presidio anonymization for inputs, a zero-shot topic guard to enforce scope, an output safety checker that uses Groq’s Llama Guard 3 when available (with a safe local fallback), adversarial testing, and an end-to-end async pipeline with benchmarked latency breakdown and overhead vs baseline. **Phase D** provides a concise production blueprint with SLOs, an architecture diagram, an alert playbook, and an order-of-magnitude cost model.

The intent is to keep implementations minimal but real, with clear artifacts (CSV/JSON/MD) produced by each phase and a CI “eval gate” that can fail a pull request when quality drops below agreed thresholds. Total evaluation cost depends on provider pricing and token usage and should be recorded after runs.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` (copy `.env.example` to `.env` if you prefer local env files).

## Commands

```bash
python -m rag_pipeline.build_index
python -m rag_pipeline.query "What is X?"
python phase-a/generate_testset.py
python phase-a/run_eval.py --threshold faithfulness=0.85 answer_relevancy=0.80 context_precision=0.70 context_recall=0.75
python phase-a/failure_analysis.py
python phase-b/pairwise_judge.py --n 30
python phase-b/absolute_scoring.py --n 30
python phase-b/kappa_analysis.py
python phase-b/bias_report.py
python phase-c/input_guard.py
python phase-c/topic_guard.py
python phase-c/adversarial_test.py
python phase-c/output_guard.py
python phase-c/full_pipeline.py
```

## Results Summary

- **Phase A:** `phase-a/testset_v1.csv`, `phase-a/ragas_results.csv`, `phase-a/ragas_summary.json`, `phase-a/failure_analysis.md`
- **Phase B:** `phase-b/pairwise_results.csv`, `phase-b/absolute_scores.csv`, `phase-b/kappa_summary.md`, `phase-b/judge_bias_report.md`, `phase-b/judge_bias_chart.png`
- **Phase C:** `phase-c/pii_test_results.csv`, `phase-c/topic_guard_results.csv`, `phase-c/adversarial_test_results.csv`, `phase-c/output_guard_results.csv`, `phase-c/latency_benchmark.csv`, `phase-c/audit_log.jsonl`
- **Phase D:** `phase-d/blueprint.md`

Total eval cost (placeholder): **$TODO** (record provider, model(s), and run date).

Blueprint: `phase-d/blueprint.md`

## Demo Video

TODO: Add a link or path to a short demo video (index → query → eval gate → guardrail tests).

## Lessons Learned

- Prefer evaluation gates early: failing fast in CI prevents silent regressions.
- Track both quality and ops: metric thresholds plus latency/overhead benchmarks reveal tradeoffs.
- Guardrails need measurement: adversarial detection and false-positive rates should be tracked like any other metric.
