# Project Report

This folder contains the final research paper for the project:

**Token-Generation Latency Benchmarking in LLaMA**

## Contents

- `Team_15_LLMSystems_paper.pdf`  
  Final report describing the methodology, experiments, results, and analysis.

## Description

The paper presents a benchmarking study of token-generation latency in LLaMA-family models using TinyLlama as a proxy. It analyzes how prompt length and output length affect:

- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- End-to-End Latency
- Throughput

## Key Contributions

- Separation of prefill and decode phases in LLM inference
- Measurement of latency metrics under controlled input/output sizes
- Analysis of trade-offs between latency and throughput
- Experimental results supported by benchmark data and plots

## Notes

- The experiments were conducted on CPU using TinyLlama (1.1B).
- Results focus on latency trends rather than absolute performance.
- The report is supported by code and results available in other folders of this repository.
