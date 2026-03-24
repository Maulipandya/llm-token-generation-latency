# Benchmarking LLM Token-Generation Latency

This folder contains the implementation for benchmarking token-generation latency in LLaMA-family models using TinyLlama.

## Files

benchmark_llm_latency.py  
Runs latency experiments and records metrics for different prompt and output lengths.

plot_results.py  
Generates plots from the benchmark results CSV file.

## Metrics Measured

Time to First Token (TTFT)  
Time taken to generate the first token after processing the input prompt.

Time Per Output Token (TPOT)  
Average time required to generate each token after the first.

End-to-End Latency  
Total time required to generate the complete output sequence.

Throughput (tokens/sec)  
Number of tokens generated per second.

## Experiment Setup

Model: TinyLlama (1.1B) as a proxy for LLaMA  
Device: CPU  
Batch size: 1 (simulates real-time usage)  
Prompt lengths: 32, 128, 256, 512 tokens  
Output lengths: 32, 64, 128 tokens  
Multiple runs are averaged to reduce variance  

## Methodology

The first forward pass measures TTFT (prefill + first token).  
Remaining tokens are generated sequentially using KV-cache.  
TPOT is computed from the decode phase.  
Total latency is the sum of prefill and decode time.  

## Usage

Run Benchmark  
```
python benchmark_llm_latency.py
```

This generates:  
benchmark_results.csv  

Generate Plots  
```
python plot_results.py
```

This generates:  
ttft_vs_prompt.png  
e2e_vs_output.png  
throughput_vs_prompt.png  
tpot_vs_output.png  

## Notes

The benchmark focuses on latency trends rather than absolute performance.  
KV-cache is used to optimize decoding efficiency.  
Results reflect expected LLaMA-style inference behavior with clear separation between prefill and decode phases.
