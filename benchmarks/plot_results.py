import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("benchmark_results.csv")

# Average across repeated runs
avg = (
    df.groupby(["prompt_tokens", "output_tokens"], as_index=False)
      .agg({
          "ttft_s": "mean",
          "tpot_s": "mean",
          "total_latency_s": "mean",
          "throughput_tok_s": "mean"
      })
)

# -----------------------------
# Figure 1: TTFT vs Prompt Length
# -----------------------------
plt.figure(figsize=(7, 4.5))
for output_len in sorted(avg["output_tokens"].unique()):
    subset = avg[avg["output_tokens"] == output_len]
    plt.plot(
        subset["prompt_tokens"],
        subset["ttft_s"],
        marker="o",
        label=f"Output = {output_len}"
    )

plt.xlabel("Prompt Length (tokens)")
plt.ylabel("TTFT (seconds)")
plt.title("TTFT vs Prompt Length")
plt.xticks(sorted(avg["prompt_tokens"].unique()))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("ttft_vs_prompt.png", dpi=300)
plt.close()

# -----------------------------
# Figure 2: End-to-End Latency vs Output Length
# -----------------------------
plt.figure(figsize=(7, 4.5))
for prompt_len in sorted(avg["prompt_tokens"].unique()):
    subset = avg[avg["prompt_tokens"] == prompt_len]
    plt.plot(
        subset["output_tokens"],
        subset["total_latency_s"],
        marker="o",
        label=f"Prompt = {prompt_len}"
    )

plt.xlabel("Output Length (tokens)")
plt.ylabel("End-to-End Latency (seconds)")
plt.title("End-to-End Latency vs Output Length")
plt.xticks(sorted(avg["output_tokens"].unique()))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("e2e_vs_output.png", dpi=300)
plt.close()

# -----------------------------
# Figure 3: Throughput vs Prompt Length
# -----------------------------
plt.figure(figsize=(7, 4.5))
for output_len in sorted(avg["output_tokens"].unique()):
    subset = avg[avg["output_tokens"] == output_len]
    plt.plot(
        subset["prompt_tokens"],
        subset["throughput_tok_s"],
        marker="o",
        label=f"Output = {output_len}"
    )

plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs Prompt Length")
plt.xticks(sorted(avg["prompt_tokens"].unique()))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("throughput_vs_prompt.png", dpi=300)
plt.close()

# -----------------------------
# Figure 4: TPOT vs Output Length
# -----------------------------
plt.figure(figsize=(7, 4.5))
for prompt_len in sorted(avg["prompt_tokens"].unique()):
    subset = avg[avg["prompt_tokens"] == prompt_len]
    plt.plot(
        subset["output_tokens"],
        subset["tpot_s"],
        marker="o",
        label=f"Prompt = {prompt_len}"
    )

plt.xlabel("Output Length (tokens)")
plt.ylabel("TPOT (seconds/token)")
plt.title("TPOT vs Output Length")
plt.xticks(sorted(avg["output_tokens"].unique()))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("tpot_vs_output.png", dpi=300)
plt.close()

print("Plots generated successfully.")