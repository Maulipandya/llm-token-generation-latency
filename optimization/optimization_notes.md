# Optimization Notes

## Potential Improvements

### 1. GPU Acceleration
Running the model on GPU can significantly reduce TTFT and TPOT compared to CPU execution.

### 2. Quantization
Using lower precision models (int8 or 4-bit) can reduce memory usage and improve inference speed.

### 3. Batching
Processing multiple requests together can improve throughput, although it may increase TTFT.

### 4. KV-Cache Optimization
Efficient use of KV-cache reduces redundant computation during decoding and improves TPOT.

### 5. Model Size Selection
Smaller models (e.g., TinyLlama) provide faster inference, while larger models improve output quality but increase latency.

---

## Goal

The goal of optimization is to:
- Reduce Time to First Token (TTFT)
- Improve decoding speed (TPOT)
- Maintain a balance between performance and model quality
