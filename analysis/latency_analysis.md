# Latency Analysis

## Observations
- Time to First Token (TTFT) increases as prompt length increases because more computation is required during the prefill phase.
- End-to-end latency increases with output length because tokens are generated sequentially during decoding.
- Time Per Output Token (TPOT) remains relatively stable across output lengths, showing the benefit of KV-cache during decoding.
- Throughput varies across configurations, but higher throughput does not always mean better user-perceived responsiveness.
- The results show a clear separation between prefill cost and decode cost in LLaMA inference.

## Findings
- Prompt length primarily affects TTFT.
- Output length primarily affects end-to-end latency.
- TPOT is more stable than TTFT across different prompt lengths.
- KV-cache helps maintain decoding efficiency after the first token.
- Throughput alone is not sufficient to evaluate interactive performance.

## Interpretation
LLM inference consists of two main phases:

### Prefill Phase
- Processes the full input prompt
- Dominates Time to First Token (TTFT)
- Becomes more expensive as prompt length increases

### Decode Phase
- Generates output tokens one by one
- Determines TPOT and total latency
- Benefits from KV-cache reuse

## Conclusion
- TTFT is critical for responsiveness in real-time applications.
- Output length has a strong effect on total completion time.
- TPOT stays comparatively stable because decoding reuses cached attention states.
- Both latency and throughput should be considered when evaluating LLM inference performance.
