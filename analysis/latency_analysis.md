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

## Findings

- Prompt length mainly affects **Time to First Token (TTFT)**.
- Output length mainly affects **total latency**.
- **KV-cache improves decoding efficiency**, keeping TPOT stable.
- **Latency metrics are more important than throughput** for real-time applications.
- There is a clear separation between **prefill** and **decode** phases.

---

## Interpretation

The results show two phases in LLM inference:

### Prefill Phase
- Depends on prompt length
- Affects TTFT

### Decode Phase
- Depends on output length
- Affects TPOT and total latency

---

## Conclusion

- TTFT is critical for responsiveness
- Output length affects total latency
- Both latency and throughput should be considered
