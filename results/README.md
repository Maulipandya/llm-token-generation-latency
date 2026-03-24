# Results

This folder contains the output of latency benchmarking experiments for LLaMA-family models.

## Contents

- `benchmark_results.csv`  
  Contains measured metrics for different prompt and output lengths.

## Metrics Included

- **TTFT (Time to First Token)**  
  Time taken to generate the first token after input.

- **TPOT (Time Per Output Token)**  
  Average time per generated token after the first.

- **End-to-End Latency**  
  Total time to generate the complete output.

- **Throughput (tokens/sec)**  
  Number of tokens generated per second.

## Notes

- Results are averaged over multiple runs for consistency.
- Experiments are conducted with batch size = 1 to simulate real-time scenarios.
- These results support the analysis and findings presented in the project report.
