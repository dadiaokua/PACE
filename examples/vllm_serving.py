#!/usr/bin/env python3
"""Minimal vLLM + PACE integration example.

Prerequisites:
  - NVIDIA GPU with clock-control permission (pynvml or nvidia-smi -lgc)
  - vLLM installed in your serving environment
  - A local or remote model path

Usage:
  python examples/vllm_serving.py --model /path/to/model --gpu-indices 0
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM with PACE DVFS enabled")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--gpu-indices", default="0", help="Comma-separated GPU ids")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--slo-margin", type=float, default=1.2)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed-cost weight alpha in the workload score (default: 0.35)",
    )
    args = parser.parse_args()

    gpu_indices = [int(x.strip()) for x in args.gpu_indices.split(",") if x.strip()]

    from vllm import LLM, SamplingParams

    from pace import install_pace

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    uninstall = install_pace(
        llm.llm_engine,
        gpu_index=gpu_indices[0],
        gpu_indices=gpu_indices,
        slo_margin=args.slo_margin,
        fixed_cost_weight=args.alpha,
    )

    prompts = [
        "Explain why continuous batching makes batch size a weak load proxy.",
        "Summarize KV-cache memory traffic during LLM decode.",
    ]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))

    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text.strip()
        print(f"\nPrompt: {prompt}\nAnswer: {text[:200]}{'...' if len(text) > 200 else ''}")

    stats = uninstall()
    print("\nPACE stats:", stats)


if __name__ == "__main__":
    main()
