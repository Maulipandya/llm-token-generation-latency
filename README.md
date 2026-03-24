# Token-Generation Latency Benchmarking in LLaMA

## Overview
This project benchmarks token-generation latency in LLaMA-family models using TinyLlama as a proxy.  
The goal is to analyze how prompt length and output length affect real-time inference performance.

---

## Metrics
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- End-to-End Latency
- Throughput (tokens/sec)

---

## Key Findings
- **TTFT increases with prompt length** due to higher prefill computation.
- **End-to-end latency increases with output length** because tokens are generated sequentially.
- **TPOT remains relatively stable**, showing efficiency of KV-cache during decoding.
- **Throughput alone is not sufficient** to evaluate performance; latency must also be considered.

---

## How the Paper is Implemented in Code

The implementation directly follows the concepts described in the paper:

- **Prefill vs Decode Separation**
  - TTFT is measured using the first forward pass (prompt processing + first token).
  - Remaining tokens are generated step-by-step to simulate autoregressive decoding.

- **Controlled Input Sizes**
  - Prompts are generated with exact token lengths (32, 128, 256, 512).
  - Output lengths are fixed (32, 64, 128) for consistent comparison.

- **Metric Computation**
  - TTFT → Time until first token
  - TPOT → Average time per generated token
  - Total Latency → TTFT + decode time
  - Throughput → Tokens generated per second

- **Experimental Consistency**
  - Batch size = 1 (real-time scenario)
  - Multiple runs averaged to reduce variance

---

## Setup & Usage
```bash
pip install torch transformers
python benchmark.py

Notes
TinyLlama (1.1B) is used as a proxy for LLaMA due to hardware constraints.
Experiments are performed on CPU.
Results capture trends consistent with LLaMA-based systems.
