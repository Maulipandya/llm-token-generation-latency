# Token-Generation Latency Benchmarking in LLaMA

## Overview
This project benchmarks token-generation latency in LLaMA-family models using **TinyLlama** as a lightweight proxy.  
The goal is to analyze how **prompt length** and **output length** affect real-time inference performance in autoregressive text generation.

---

## Metrics

The following latency metrics are evaluated:

- **Time to First Token (TTFT)** – Time taken to generate the first token after input.
- **Time Per Output Token (TPOT)** – Average time to generate each token after the first.
- **End-to-End Latency** – Total time to generate the complete output.
- **Throughput (tokens/sec)** – Number of tokens generated per second.

---

## Key Findings

- **TTFT increases with prompt length** due to higher computation during the prefill phase.
- **End-to-end latency increases with output length** because tokens are generated sequentially.
- **TPOT remains relatively stable**, demonstrating the efficiency of KV-cache during decoding.
- **Throughput alone is not sufficient** to evaluate performance; latency must also be considered.

---

## How the Paper is Implemented in Code

The implementation directly follows the methodology described in the paper:

### 1. Prefill vs Decode Separation
- TTFT is measured using the first forward pass (prompt processing + first token).
- Remaining tokens are generated step-by-step to simulate autoregressive decoding.

### 2. Controlled Input Sizes
- Prompts are generated with fixed token lengths: **32, 128, 256, 512**
- Output lengths are fixed: **32, 64, 128**

### 3. Metric Computation
- **TTFT** → Time until first token  
- **TPOT** → Average time per generated token  
- **Total Latency** → TTFT + decode time  
- **Throughput** → Tokens generated per second  

### 4. Experimental Consistency
- Batch size = 1 (simulates real-time usage)
- Multiple runs averaged to reduce variance

---

## Repository Structure

```text
analysis/        → Interpretation and findings
benchmarks/      → Benchmarking code
optimization/    → Performance improvement ideas
report/          → Research paper
results/         → Benchmark outputs (CSV)
