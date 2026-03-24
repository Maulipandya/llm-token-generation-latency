import os
import csv
import time
import argparse
import statistics
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TTFT, TPOT, total latency, and throughput for LLaMA-family models.")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Hugging Face model name. For paper match, replace with your actual LLaMA-family model.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to run on.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "float32"],
                        help="Model dtype.")
    parser.add_argument("--runs", type=int, default=5, help="Number of measured runs per configuration.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs.")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[32, 128, 256, 512],
                        help="Prompt lengths in tokens.")
    parser.add_argument("--output_lengths", type=int, nargs="+", default=[32, 64, 128],
                        help="Output lengths in tokens.")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
                        help="CSV file for detailed results.")
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def pick_dtype(dtype_arg: str, device: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def exact_token_prompt(tokenizer, target_tokens: int) -> str:
    """
    Build text whose tokenized length is exactly target_tokens.
    """
    base = (
        "Large language models are useful for studying latency, throughput, "
        "prefill, and decode performance in interactive systems. "
    )
    text = base

    while True:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_tokens:
            ids = ids[:target_tokens]
            return tokenizer.decode(ids, skip_special_tokens=True)
        text += base


@torch.no_grad()
def measure_latency(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: str,
) -> Tuple[float, float, float, float]:
    """
    Returns:
        TTFT, TPOT, total_latency, throughput
    """
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Measure TTFT = prefill + first token
    sync_device(device)
    start_ttft = time.perf_counter()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    first_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = out.past_key_values

    sync_device(device)
    ttft = time.perf_counter() - start_ttft

    # Decode remaining tokens one by one
    remaining = max_new_tokens - 1
    decode_time = 0.0

    if remaining > 0:
        current_input_ids = first_token
        current_attention_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=device,
        )

        sync_device(device)
        start_decode = time.perf_counter()

        for _ in range(remaining):
            step_out = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            past_key_values = step_out.past_key_values
            current_input_ids = next_token

        sync_device(device)
        decode_time = time.perf_counter() - start_decode

    total_latency = ttft + decode_time
    generated_tokens = max_new_tokens

    if generated_tokens > 1:
        tpot = decode_time / (generated_tokens - 1)
    else:
        tpot = 0.0

    throughput = generated_tokens / total_latency if total_latency > 0 else 0.0
    return ttft, tpot, total_latency, throughput


def mean_std(values: List[float]) -> Tuple[float, float]:
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


def print_header(args, device, dtype) -> None:
    print("\n=== LLM Latency Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Runs per config: {args.runs}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Output lengths: {args.output_lengths}")
    print("============================\n")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    print_header(args, device, dtype)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    results: List[Dict] = []

    # Warmup
    print("Running warmup...")
    warmup_prompt = exact_token_prompt(tokenizer, 32)
    for _ in range(args.warmup):
        _ = measure_latency(model, tokenizer, warmup_prompt, 16, device)

    print("\n=== Benchmark Results ===\n")

    for prompt_len in args.prompt_lengths:
        prompt_text = exact_token_prompt(tokenizer, prompt_len)
        actual_prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        for output_len in args.output_lengths:
            ttfts, tpots, totals, throughputs = [], [], [], []

            for run_id in range(1, args.runs + 1):
                ttft, tpot, total, thr = measure_latency(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=output_len,
                    device=device,
                )

                ttfts.append(ttft)
                tpots.append(tpot)
                totals.append(total)
                throughputs.append(thr)

                results.append({
                    "model": args.model,
                    "device": device,
                    "prompt_tokens": actual_prompt_len,
                    "output_tokens": output_len,
                    "run_id": run_id,
                    "ttft_s": round(ttft, 6),
                    "tpot_s": round(tpot, 6),
                    "total_latency_s": round(total, 6),
                    "throughput_tok_s": round(thr, 6),
                })

            ttft_mean, ttft_std = mean_std(ttfts)
            tpot_mean, tpot_std = mean_std(tpots)
            total_mean, total_std = mean_std(totals)
            thr_mean, thr_std = mean_std(throughputs)

            print(f"Prompt: {actual_prompt_len} | Output: {output_len}")
            print(f"TTFT:            {ttft_mean:.4f}s ± {ttft_std:.4f}")
            print(f"TPOT:            {tpot_mean:.4f}s ± {tpot_std:.4f}")
            print(f"Total Latency:   {total_mean:.4f}s ± {total_std:.4f}")
            print(f"Throughput:      {thr_mean:.2f} tok/s ± {thr_std:.2f}")
            print("-" * 60)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved detailed results to: {args.output_csv}")


if __name__ == "__main__":
    main()