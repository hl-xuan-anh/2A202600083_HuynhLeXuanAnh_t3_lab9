# Failure Analysis (Phase A)

## Aggregate Metrics (Mean)

- faithfulness: 0.838
- answer_relevancy: 0.980
- context_precision: 0.930
- context_recall: 0.989

## Bottom 10 Failures (by Avg)

| # | Question | Type | F | AR | CP | CR | Avg | Cluster |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | What are the perks of an AI retrieval layer for accuracy? | reasoning | 0.455 | 1.000 | 0.000 | 1.000 | 0.614 | C4 multi-hop/reasoning |
| 2 | What role do embeddings play in modern AI architecture for latency and retrieva… | multi_context | 1.000 | 0.964 | 0.333 | 1.000 | 0.824 | C4 multi-hop/reasoning |
| 3 | What key strategies and metrics optimize latency in enterprise AI, especially f… | multi_context | 0.455 | 0.946 | 1.000 | 1.000 | 0.850 | C4 multi-hop/reasoning |
| 4 | What factors balance for optimal AI latency? | reasoning | 1.000 | 0.990 | 0.500 | 1.000 | 0.872 | C4 multi-hop/reasoning |
| 5 | What is the role of document chunking in the architecture of semantic search in… | simple | 0.583 | 0.995 | 1.000 | 1.000 | 0.895 | C2 faithfulness/hallucination |
| 6 | What do engineers consider for cost, reliability, and scalability in AI? | reasoning | 0.667 | 0.924 | 1.000 | 1.000 | 0.898 | C4 multi-hop/reasoning |
| 7 | What key strategies and metrics optimize reranking in enterprise AI? | multi_context | 0.615 | 0.981 | 1.000 | 1.000 | 0.899 | C4 multi-hop/reasoning |
| 8 | What is the purpose of guardrail systems in enterprise AI deployments? | simple | 0.600 | 1.000 | 1.000 | 1.000 | 0.900 | C2 faithfulness/hallucination |
| 9 | How does integrating external knowledge via retrieval layers improve AI respons… | multi_context | 0.625 | 0.975 | 1.000 | 1.000 | 0.900 | C4 multi-hop/reasoning |
| 10 | How do eval metrics help balance cost, reliability, and scalability in AI with … | multi_context | 0.667 | 0.938 | 1.000 | 1.000 | 0.901 | C4 multi-hop/reasoning |

## Clusters & Recommendations

## C4 multi-hop/reasoning
**Pattern:** Multi-context/reasoning questions often need evidence across multiple chunks.
**Examples:**
- What are the perks of an AI retrieval layer for accuracy?
- What role do embeddings play in modern AI architecture for latency and retrieval?
- What key strategies and metrics optimize latency in enterprise AI, especially for multi-hop retrieval and augmented gen…
**Root cause:** Single-pass retrieval misses supporting steps; contexts are fragmented.
**Proposed technical fix:** Increase `top_k`, add multi-query retrieval, or iterative retrieval (retrieve → draft → retrieve again).

## C2 faithfulness/hallucination
**Pattern:** Faithfulness low even when some context exists → model may be inventing details not supported by retrieved text.
**Examples:**
- What is the role of document chunking in the architecture of semantic search in enterprise AI systems?
- What is the purpose of guardrail systems in enterprise AI deployments?
**Root cause:** Prompt does not strongly constrain the model; retrieved contexts too long/irrelevant; missing citations requirement.
**Proposed technical fix:** Tighten prompt ("use only contexts" + refusal), trim/format contexts, and add a post-check to refuse when evidence is weak.

