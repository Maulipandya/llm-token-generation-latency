# Latency Analysis

## Observations

- **TTFT increases with prompt length**
  - Longer prompts require more computation during the prefill phase.

- **End-to-end latency increases with output length**
  - Because tokens are generated sequentially in autoregressive decoding.

- **TPOT remains relatively stable**
  - Due to KV-cache reuse during decoding.

- **Throughput improves for larger outputs**
  - But does not fully represent user-perceived latency.

---

## Interpretation

The results clearly show two phases in LLM inference:

### 1. Prefill Phase
- Depends on prompt length
- Dominates TTFT

### 2. Decode Phase
- Depends on output length
- Controls TPOT and total latency

---

## Conclusion

- TTFT is critical for real-time applications
- Output length mainly affects total response time
- Both latency and throughput must be considered for performance evaluation
