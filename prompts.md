# Prompts (Chronological)

Note: AI-generated code was reviewed and modified before commit (naming, robustness, error handling, and lab-specific constraints).

## 1) Repo scaffold for phased evaluation

1. “Create the exact repository structure below… only create folders, placeholder files, minimal TODO comments… add short README skeleton…"

## 2) Minimal LangChain RAG (docs → vectorstore)

2. “Add a minimal but real RAG pipeline… place it in `rag_pipeline/`… use LangChain + OpenAIEmbeddings + Chroma/FAISS… `gpt-4o-mini`… expose sync/async… build_index and query modules… wire Phase A/Phase C imports… update README commands…"
3. “Fill requirements.txt for a Python 3.10+ RAG evaluation and guardrails lab… pinned versions…"

## 3) Phase A: testset generation, eval gate, failure analysis

4. “Implement `phase-a/generate_testset.py`… DirectoryLoader(`./docs`, glob=`**/*.md`)… RAGAS TestsetGenerator… distribution… save `testset_v1.csv`… print `evolution_type` counts…"
5. “Implement `phase-a/run_eval.py`… run my RAG pipeline per question… evaluate with RAGAS… save `ragas_results.csv` and `ragas_summary.json`… thresholds… exit code 1… `gpt-4o-mini`…"
6. “Implement `phase-a/failure_analysis.py`… bottom 10… heuristic clusters… write `failure_analysis.md`…"
7. “Create `.github/workflows/eval-gate.yml`… PR to main… run eval gate… upload artifacts always…"

## 4) Phase B: LLM-as-judge evaluations

8. “Implement `phase-b/pairwise_judge.py`… LLM judge JSON only… swap-and-average… save `pairwise_results.csv`…"
9. “Implement `phase-b/absolute_scoring.py`… 4 dimensions… JSON parsing… save `absolute_scores.csv`…"
10. “Implement `phase-b/kappa_analysis.py`… Cohen’s kappa… save `kappa_summary.md`…"
11. “Implement `phase-b/bias_report.py`… position bias + length bias… matplotlib chart… markdown mitigations…"

## 5) Phase C: guardrails, adversarial tests, full pipeline

12. “Implement `phase-c/input_guard.py`… VN PII regex + Presidio anonymization… test runner… save `pii_test_results.csv`… detection rate + P95 latency…"
13. “Implement `TopicGuard`… LLM zero-shot… 20 tests… save `topic_guard_results.csv`…"
14. “Implement `phase-c/adversarial_test.py`… 20 adversarial + 10 legit… run sanitize + topic guard… save results… print detection/false positive rates…"
15. “Implement `phase-c/output_guard.py`… Groq Llama Guard 3 with fallback if `GROQ_API_KEY` missing… test harness… save `output_guard_results.csv`…"
16. “Implement `phase-c/full_pipeline.py`… async guarded pipeline L1–L4… benchmark 100 requests… baseline overhead… save `latency_benchmark.csv`…"

## 6) Phase D and final documentation

17. “Write `phase-d/blueprint.md`… SLOs, Mermaid diagram, incidents, cost analysis…"
18. “Finalize `README.md` and `prompts.md`"
