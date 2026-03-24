# Token-Generation Latency Benchmarking in LLaMA

## Overview
This project benchmarks token-generation latency in LLaMA-family models using TinyLlama as a lightweight proxy. The goal is to analyze how prompt length and output length affect real-time inference performance in autoregressive text generation.

## Metrics
The following latency metrics are evaluated:

- **Time to First Token (TTFT)** – Time taken to generate the first token after input.
- **Time Per Output Token (TPOT)** – Average time to generate each token after the first.
- **End-to-End Latency** – Total time to generate the complete output.
- **Throughput (tokens/sec)** – Number of tokens generated per second.

## Key Findings
- TTFT increases with prompt length due to higher computation during the prefill phase.
- End-to-end latency increases with output length because tokens are generated sequentially.
- TPOT remains relatively stable, demonstrating the efficiency of KV-cache during decoding.
- Throughput alone is not sufficient to evaluate performance; latency must also be considered.

## How the Paper is Implemented in Code

### Prefill vs Decode Separation
- TTFT is measured using the first forward pass (prompt processing + first token).
- Remaining tokens are generated step-by-step to simulate autoregressive decoding.

### Controlled Input Sizes
- Prompt lengths: **32, 128, 256, 512 tokens**
- Output lengths: **32, 64, 128 tokens**

### Metric Computation
- TTFT → Time until first token  
- TPOT → Average time per generated token  
- Total Latency → TTFT + decode time  
- Throughput → Tokens generated per second  

### Experimental Consistency
- Batch size = 1 (simulates real-time usage)
- Multiple runs averaged to reduce variance

## Repository Structure
- `analysis/` → Interpretation and findings  
- `benchmarks/` → Benchmarking and plotting code  
- `optimization/` → Performance improvement ideas  
- `report/` → Research paper  
- `results/` → Benchmark outputs (CSV)

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch transformers matplotlib pandas
```

## Usage

Run the benchmark:
```bash
python benchmarks/benchmark_llm_latency.py
```

Generate plots:
```bash
python benchmarks/plot_results.py
```

## Results
Benchmark outputs are stored in:
```
results/benchmark_results.csv
```

These results support the analysis and findings presented in the project report.

## Interpretation of Results

The reported results reflect latency trends observed under controlled CPU-based experiments using a LLaMA-family model.

Key observations supported by the data:
- TTFT increases with prompt length due to prefill cost
- End-to-end latency increases with output length due to sequential decoding
- TPOT remains relatively stable across configurations
- Throughput alone does not represent user-perceived responsiveness

These findings align with the theoretical behavior of autoregressive transformer inference and are intended to highlight relative performance characteristics rather than absolute deployment benchmarks.

## Notes
- TinyLlama (1.1B) is used as a proxy for LLaMA due to hardware constraints.
- Experiments are performed on CPU.
- The benchmark focuses on latency trends rather than absolute performance.
- Results are consistent with expected behavior of LLaMA-style models.

## Author
Team 15 – LLMSystems
