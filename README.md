# Token-Generation Latency Benchmarking in LLaMA

## Overview
This project benchmarks token-generation latency in LLaMA-family models using TinyLlama as a lightweight proxy. The goal is to analyze how prompt length and output length affect real-time inference performance in autoregressive text generation.

## Metrics
The following latency metrics are evaluated:
- Time to First Token (TTFT) – Time taken to generate the first token after input.
- Time Per Output Token (TPOT) – Average time to generate each token after the first.
- End-to-End Latency – Total time to generate the complete output.
- Throughput (tokens/sec) – Number of tokens generated per second.

## Key Findings
- TTFT increases with prompt length due to higher computation during the prefill phase.
- End-to-end latency increases with output length because tokens are generated sequentially.
- TPOT remains relatively stable, demonstrating the efficiency of KV-cache during decoding.
- Throughput alone is not sufficient to evaluate performance; latency must also be considered.

## How the Paper is Implemented in Code
The implementation directly follows the methodology described in the paper:

Prefill vs Decode Separation
- TTFT is measured using the first forward pass (prompt processing + first token).
- Remaining tokens are generated step-by-step to simulate autoregressive decoding.

Controlled Input Sizes
- Prompts are generated with fixed token lengths: 32, 128, 256, 512
- Output lengths are fixed: 32, 64, 128

Metric Computation
- TTFT → Time until first token
- TPOT → Average time per generated token
- Total Latency → TTFT + decode time
- Throughput → Tokens generated per second

Experimental Consistency
- Batch size = 1 (simulates real-time usage)
- Multiple runs averaged to reduce variance

## Repository Structure
analysis/        → Interpretation and findings  
benchmarks/      → Benchmarking code  
optimization/    → Performance improvement ideas  
report/          → Research paper  
results/         → Benchmark outputs (CSV)  

## Setup
Install dependencies:
pip install -r requirements.txt

Or manually:
pip install torch transformers

## Usage
Run the benchmark:
python benchmarks/benchmark_llm_latency.py

## Results
Benchmark outputs are stored in:
results/benchmark_results.csv

These results support the analysis and findings presented in the project report.

## Notes
- TinyLlama (1.1B) is used as a proxy for LLaMA due to hardware constraints.
- Experiments are performed on CPU.
- The benchmark focuses on latency trends rather than absolute performance.
- Results are consistent with expected behavior of LLaMA-style models.

## Author
Team 15 – LLMSystems
