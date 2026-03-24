# Latency Analysis

## Observations

- Time to First Token (TTFT) increases as prompt length increases because more computation is required during the prefill phase.
- End-to-end latency increases with output length since tokens are generated sequentially.
- Time Per Output Token (TPOT) remains relatively stable due to the use of KV-cache.
- Throughput improves for larger workloads but does not fully reflect user-perceived latency.

---

## Findings

- Prompt length primarily affects TTFT.
- Output length primarily affects total latency.
- KV-cache helps maintain stable decoding performance.
- Throughput alone is not sufficient to evaluate real-time performance.
- There is a clear separation between prefill and decode phases.

---

## Interpretation

LLM inference consists of two main phases:

### Prefill Phase
- Processes the input prompt
- Dominates TTFT

### Decode Phase
- Generates tokens one by one
- Determines TPOT and total latency

---

## Conclusion

- TTFT is critical for responsiveness in real-time applications.
- Output length impacts total response time.
- Both latency and throughput should be considered for evaluating performance.
