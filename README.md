# PACE

**PACE** (Position-Aware Control for Energy-efficient serving) is a lightweight, feedforward GPU DVFS controller for LLM serving under **continuous batching**.

Instead of using active batch size alone, PACE reads the **active KV working set**

\[
S_{\mathrm{KV}}(t) = \sum_{i=1}^{B(t)} p_i(t)
\]

where \(p_i(t)\) is request \(i\)'s current token position. This signal is available from the scheduler **before** each decode step and tracks decode-time load more accurately than batch size when requests are at different context depths.

PACE maps \((B, S_{\mathrm{KV}})\) to a workload score and selects a clock level on the GPU frequency ladder under a tail-latency margin, with prefill bypass and throughput guards.

## Install

```bash
git clone git@github.com:dadiaokua/PACE.git
cd PACE
pip install -e .
```

Optional runtime dependency for real DVFS actuation:

```bash
pip install nvidia-ml-py3
```

vLLM is required only for the serving integration example (`pip install vllm` in your environment).

## Quick start (vLLM)

```python
from vllm import LLM, SamplingParams
from pace import install_pace

llm = LLM(model="Qwen/Qwen3-8B", max_model_len=8192)

uninstall = install_pace(
    llm.llm_engine,
    gpu_index=0,
    gpu_indices=[0],      # tensor-parallel group
    slo_margin=1.2,
    fixed_cost_weight=0.35,
)

outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
stats = uninstall()
print(stats)
```

Or run the bundled example:

```bash
python examples/vllm_serving.py --model /path/to/model --gpu-indices 0
```

## How it works

1. **Hook**: patch `engine.scheduler[0]._schedule` so PACE runs immediately before each scheduling step.
2. **Signal**: read active batch size \(B\) and token positions from the running queue; compute \(S_{\mathrm{KV}} = \sum_i p_i\).
3. **Score**:  
   `load = alpha * (B / B_max) + (1 - alpha) * (S_KV / S_max)`
4. **Actuation**: map `load / slo_margin` to the lowest safe frequency on the discrete GPU clock ladder.
5. **Guards**: force max clock during prefill; tighten margin if throughput drops; optional online calibration of `B_max` and `S_max` during warmup.

The default `alpha = 0.35` corresponds to the fixed weight-read term in the linear energy law \(E_{\mathrm{step}} = C + k S_{\mathrm{KV}}\). Override via `fixed_cost_weight` or the `PACE_ALPHA` environment variable.

## GPU clock permissions

Real frequency scaling requires permission to lock clocks:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -i 0 -lgc 135,1530   # example range; use your GPU's supported ladder
```

Without permissions, PACE falls back to **record-only** mode: it computes target frequencies but does not change clocks.

## Project layout

```
pace/
  config.py         # PACEConfig and load/frequency mapping
  gpu.py            # NVML / nvidia-smi frequency controller
  working_set.py    # S_KV extraction from vLLM scheduler state
  vllm_plugin.py    # install_pace() hook for vLLM
examples/
  vllm_serving.py   # minimal end-to-end demo
```

## Citation

If you use PACE in academic work, please cite the ATC 2026 paper (link to be added upon publication):

```bibtex
@inproceedings{pace2026,
  title={PACE: Turning the KV Working Set into a Runtime Signal for Energy-Efficient LLM Serving},
  booktitle={Proceedings of the ACM SIGOPS Annual Technical Conference (ATC)},
  year={2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
