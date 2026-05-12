# Phase D Blueprint

## 1) SLO Definition

Scope: this lab’s end-to-end pipeline (`phase-c/full_pipeline.py`) with L1 input guards, L2 RAG, L3 output guard, L4 audit log.

| SLO | Target | Alert Threshold | Severity | Notes |
|---|---:|---:|---|---|
| **SLO-1 Faithfulness (RAGAS)** | ≥ **0.85** | < **0.85** for 1 eval run | SEV-2 | Gated in CI via `phase-a/run_eval.py --threshold ...` |
| **SLO-2 Answer Relevancy (RAGAS)** | ≥ **0.80** | < **0.80** for 1 eval run | SEV-2 | Prevents drift/rambling answers |
| **SLO-3 Retrieval Quality (Context Precision)** | ≥ **0.70** | < **0.70** for 1 eval run | SEV-2 | Detects noisy retrieval / chunking issues |
| **SLO-4 Retrieval Quality (Context Recall)** | ≥ **0.75** | < **0.75** for 1 eval run | SEV-2 | Detects missing evidence / top_k issues |
| **SLO-5 Latency (P95 total)** | ≤ **2500 ms** | > **3000 ms** for 10 min | SEV-1 | Measured by `phase-c/latency_benchmark.csv` / production telemetry |
| **SLO-6 Guardrail False Positive Rate** | ≤ **3%** | > **5%** for 30 min | SEV-2 | From `phase-c/adversarial_test_results.csv` / real traffic sampling |
| **SLO-7 Output Safety Detection Rate** | ≥ **90%** | < **85%** on nightly set | SEV-2 | From `phase-c/output_guard_results.csv` |

Operational definitions:
- RAGAS SLOs are measured on the synthetic testset (`phase-a/testset_v1.csv`) and tracked per CI run.
- Latency SLO uses request timing breakdown: **L1**, **L2**, **L3**, and **total**.
- Guardrail FP rate uses a labeled small canary set (10–50 curated queries) + periodic manual review.

## 2) Architecture Diagram

```mermaid
flowchart TD
  U[User] -->|user_input| L1[L1 Input Guards<br/>PII scrub + TopicGuard]
  L1 -->|sanitized_input| L2[L2 RAG LLM<br/>Retriever + Generator]
  L2 -->|draft_answer| L3[L3 Output Guard<br/>Groq Llama Guard 3<br/>(fallback keywords)]
  L3 -->|final_answer| U
  L1 -.->|audit event| L4[L4 Audit Log<br/>JSONL append]
  L2 -.->|audit event| L4
  L3 -.->|audit event| L4
  L4 --> S[(Storage)]

  subgraph Data
    D[./docs Markdown Corpus]
    VS[./vectorstore Local Vector DB]
  end

  D --> L2
  VS --> L2
```

Key lab components:
- L1: `phase-c/input_guard.py`, `phase-c/topic_guard.py`
- L2: `rag_pipeline/build_index.py`, `rag_pipeline/query.py`
- L3: `phase-c/output_guard.py` (Groq optional, fallback always available)
- L4: `phase-c/full_pipeline.py` audit log to `phase-c/audit_log.jsonl`

## 3) Alert Playbook

### Incident A: Faithfulness drop
- **Signal:** CI gate fails (`faithfulness < 0.85`) or trend regression across runs.
- **Immediate checks (10–20 min):**
  - Inspect `phase-a/ragas_summary.json` and bottom rows in `phase-a/ragas_results.csv`.
  - Run `python phase-a/failure_analysis.py` to cluster failures.
- **Likely causes:**
  - Retrieval mismatch (poor chunks or missing docs), prompting too permissive, context too long/noisy.
- **Mitigation:**
  - Increase retrieval quality first: adjust chunking, raise `top_k`, add reranking/hybrid retrieval.
  - Tighten generation prompt: “use ONLY contexts; otherwise say I don’t know.”
  - Add “insufficient grounding → refuse” rule before generation.
- **Follow-up:**
  - Add new testcases to `phase-a/testset_v1.csv` for the newly observed failure mode.

### Incident B: Latency spike
- **Signal:** P95 total > 3000 ms for 10 minutes; L2 dominates; or sudden L1/L3 growth.
- **Immediate checks:**
  - Compare L1/L2/L3 percentiles from `phase-c/latency_benchmark.csv` (or prod metrics).
  - Validate vectorstore load/persist behavior; check external API availability (OpenAI/Groq).
- **Likely causes:**
  - Cold starts (vectorstore load), excessive chunk counts, model rate limiting, network/API slowness.
- **Mitigation:**
  - Cache vectorstore handle; avoid re-instantiating clients per request.
  - Cap context length; reduce chunk size/count; parallelize guard checks where safe.
  - Add timeouts + circuit breaker: on slow L3, degrade to fallback output guard.

### Incident C: Guardrail false positives
- **Signal:** FP rate > 5% over 30 minutes (or repeated user complaints).
- **Immediate checks:**
  - Review `phase-c/topic_guard_results.csv` / adversarial test results and sample blocked real queries.
  - Identify patterns: specific terms triggering PII regex, over-conservative topic gating.
- **Likely causes:**
  - Over-broad regex patterns, topic classifier prompt too strict, missing allowed topics.
- **Mitigation:**
  - Narrow regex (word boundaries, contextual checks), add allowlist exceptions.
  - Update `allowed_topics` to include common valid intents for the lab.
  - Use swap-and-average or dual-LLM agreement for topic gating if needed.

## 4) Cost Analysis

Assumptions (lab-typical, conservative):
- Volume: **100,000 queries / month**
- Average tokens per query (rough):
  - **L1 TopicGuard:** ~200 input tokens, ~50 output tokens
  - **L2 RAG generation:** ~300 input tokens + ~1,200 context tokens, ~250 output tokens
  - **L3 Output guard:** ~200 input tokens, ~80 output tokens
- Total per request ≈ **(L1 250) + (L2 1,750) + (L3 280) ≈ 2,280 tokens** (input+output combined order-of-magnitude).

Monthly token volume:
- **100k * 2,280 ≈ 228M tokens/month** (order-of-magnitude estimate).

Cost model (plug in your provider pricing):
- If blended effective cost is **$X per 1M tokens**, monthly cost ≈ **228 * X**.
  - Example: at **$2 / 1M tokens** ⇒ **~$456/month**
  - Example: at **$5 / 1M tokens** ⇒ **~$1,140/month**

Non-token costs:
- Vector store storage: local disk (near-zero) for this lab; production would add managed DB/storage costs.
- Observability: logs/metrics ingestion depending on platform.

Cost reduction levers (specific to this lab):
- Reduce L1/L3 calls by caching decisions for repeated prompts or only running guards on high-risk inputs.
- Reduce context size: better retrieval (higher precision) + smaller `top_k` with reranking.
- Shorter answers (cap output tokens) and “refuse if insufficient grounding” to avoid long hallucinations.

