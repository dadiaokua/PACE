"""vLLM integration for PACE feedforward DVFS."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from pace.config import PACEConfig
from pace.gpu import GPUFrequencyController
from pace.working_set import extract_decode_state


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def install_pace(
    engine,
    gpu_index: int = 0,
    gpu_indices: Optional[List[int]] = None,
    slo_margin: float = 1.2,
    throughput_guard: float = 0.90,
    max_position: float = 12000.0,
    max_batch: int = 64,
    fixed_cost_weight: Optional[float] = None,
    verbose: bool = True,
) -> Callable[[], dict]:
    """Install PACE on a vLLM LLMEngine scheduler hook.

    Before each decode step, PACE reads the active batch size and the active KV
    working set S_KV = sum_i p_i, maps them to a workload score, and selects a
    GPU clock level feedforward (no step-time feedback required).

    Returns:
        uninstall(): restore the original scheduler hook and reset GPU clocks.
    """
    if fixed_cost_weight is None:
        alpha_env = os.environ.get("PACE_ALPHA", "").strip()
        if alpha_env:
            try:
                fixed_cost_weight = float(alpha_env)
            except ValueError:
                if verbose:
                    print(f"[PACE] ignoring invalid PACE_ALPHA={alpha_env!r}")

    config = PACEConfig(
        slo_margin=slo_margin,
        throughput_guard=throughput_guard,
        max_position=max_position,
        max_batch=max_batch,
    )
    if fixed_cost_weight is not None:
        if not 0.0 <= fixed_cost_weight <= 1.0:
            raise ValueError(f"fixed_cost_weight must be in [0, 1], got {fixed_cost_weight}")
        config.fixed_cost_weight = float(fixed_cost_weight)

    controller = GPUFrequencyController(
        gpu_index=gpu_index,
        gpu_indices=gpu_indices,
        verbose=verbose,
    )

    scheduler = engine.scheduler[0]
    original_schedule = scheduler._schedule

    step_count = [0]
    freq_history: List[Tuple[int, int, int, float, float]] = []
    total_energy_factor = [0.0]
    total_steps_with_freq = [0]
    last_freq_change_step = [0]
    load_history: List[float] = []

    active_slo_margin = [config.slo_margin]
    initial_slo_margin = [config.slo_margin]
    recent_tps: Deque[float] = deque(maxlen=config.window_size)
    warmup_tps: List[float] = []
    warmup_batch_sizes: List[int] = []
    warmup_pos_sums: List[float] = []
    baseline_throughput = [0.0]
    throttle_events = [0]
    relax_events = [0]
    last_step_time = [time.time()]
    runtime_pos_sums: Deque[float] = deque(maxlen=config.recalibrate_window)
    recalibrate_count = [0]

    def _throughput_guard_adjust() -> None:
        if baseline_throughput[0] <= 0 or len(recent_tps) < 10:
            return

        avg_recent = sum(recent_tps) / len(recent_tps)
        ratio = avg_recent / max(0.01, baseline_throughput[0])
        curr_m = active_slo_margin[0]
        init_m = initial_slo_margin[0]

        if ratio < config.throughput_guard:
            active_slo_margin[0] = max(
                init_m * config.min_slo_ratio,
                curr_m * config.tighten_factor,
            )
            throttle_events[0] += 1
        elif ratio > 1.0:
            active_slo_margin[0] = min(
                init_m * config.max_slo_ratio,
                curr_m * config.relax_factor,
            )
            relax_events[0] += 1

    def _patched_schedule():
        step_count[0] += 1
        now = time.time()
        step_dt = max(1e-6, now - last_step_time[0])
        last_step_time[0] = now

        state = extract_decode_state(scheduler.running)
        batch_size = state.batch_size
        position_sum = state.position_sum
        avg_position = state.avg_position

        if config.prefill_boost and state.has_prefill:
            target_freq = controller.max_freq
        elif step_count[0] <= config.warmup_steps:
            tps = batch_size / step_dt if batch_size > 0 else 0.0
            warmup_tps.append(tps)
            if batch_size > 0:
                warmup_batch_sizes.append(batch_size)
            if position_sum > 0:
                warmup_pos_sums.append(position_sum)

            if step_count[0] == config.warmup_steps:
                half = len(warmup_tps) // 2
                stable_tps = warmup_tps[half:] if half > 0 else warmup_tps
                baseline_throughput[0] = (
                    sum(stable_tps) / len(stable_tps) if stable_tps else 0.0
                )

                if config.auto_calibrate:
                    stable_bs = warmup_batch_sizes[len(warmup_batch_sizes) // 2 :]
                    if stable_bs:
                        obs_p95_bs = int(_percentile([float(x) for x in stable_bs], 95))
                        config.max_batch = max(16, int(obs_p95_bs * 1.5))
                    stable_ps = warmup_pos_sums[len(warmup_pos_sums) // 2 :]
                    if stable_ps:
                        obs_p95_ps = _percentile(stable_ps, 95)
                        config.max_pos_sum = max(1.0, obs_p95_ps * 1.3)

                if verbose:
                    print(
                        f"[PACE] warmup complete: baseline_tps={baseline_throughput[0]:.1f}, "
                        f"max_batch={config.max_batch}, max_pos_sum={config.max_pos_sum:.0f}"
                    )
            target_freq = controller.max_freq
        else:
            tps = batch_size / step_dt if batch_size > 0 else 0.0
            recent_tps.append(tps)
            if position_sum > 0:
                runtime_pos_sums.append(position_sum)

            target_freq = config.select_frequency(
                batch_size=batch_size,
                position_sum=position_sum,
                freq_levels=controller.freq_levels,
                slo_margin_override=active_slo_margin[0],
            )

            if step_count[0] % 5 == 0:
                _throughput_guard_adjust()

            if (
                config.auto_calibrate
                and step_count[0] % config.recalibrate_interval == 0
                and len(runtime_pos_sums) >= 50
            ):
                new_p95 = _percentile(list(runtime_pos_sums), 95)
                new_smax = max(1.0, new_p95 * 1.3)
                if new_smax > config.max_pos_sum * 1.05:
                    config.max_pos_sum = new_smax
                    recalibrate_count[0] += 1

        if (step_count[0] - last_freq_change_step[0]) < config.freq_switch_cooldown:
            target_freq = controller.current_freq

        old_freq = controller.current_freq
        actually_set = controller.set_frequency(target_freq)
        if target_freq != old_freq:
            last_freq_change_step[0] = step_count[0]

        load = config.compute_load(batch_size, position_sum)
        load_history.append(load)
        e_factor = config.energy_factor(target_freq, controller.max_freq)
        freq_history.append(
            (step_count[0], target_freq, batch_size, avg_position, position_sum)
        )
        total_energy_factor[0] += e_factor
        total_steps_with_freq[0] += 1

        if verbose and step_count[0] % 500 == 0 and step_count[0] > config.warmup_steps:
            mode = "DVFS" if actually_set else "record-only"
            print(
                f"[PACE] step={step_count[0]} freq={target_freq}MHz "
                f"batch={batch_size} sum_p={position_sum:.0f} load={load:.4f} "
                f"margin={active_slo_margin[0]:.2f} mode={mode}"
            )

        return original_schedule()

    scheduler._schedule = _patched_schedule
    engine._pace_controller = controller
    engine._pace_config = config

    if verbose:
        print("[PACE] installed")
        print(f"    backend: {controller.mode}")
        print(f"    clocks: {controller.min_freq}-{controller.max_freq} MHz")
        print(
            f"    load = {config.fixed_cost_weight:.2f}*(B/Bmax) + "
            f"{1 - config.fixed_cost_weight:.2f}*(S_KV/Smax)"
        )

    def uninstall() -> dict:
        scheduler._schedule = original_schedule
        controller.reset_frequency()

        avg_e_factor = total_energy_factor[0] / max(1, total_steps_with_freq[0])
        avg_load = sum(load_history) / len(load_history) if load_history else 0.0

        stats = {
            "pace_mode": controller.mode,
            "pace_is_real_dvfs": controller.is_real_dvfs,
            "pace_total_steps": step_count[0],
            "pace_avg_energy_factor": round(avg_e_factor, 4),
            "pace_avg_load": round(avg_load, 4),
            "pace_alpha": round(config.fixed_cost_weight, 4),
            "pace_slo_margin_initial": round(initial_slo_margin[0], 3),
            "pace_slo_margin_final": round(active_slo_margin[0], 3),
            "pace_throttle_events": throttle_events[0],
            "pace_relax_events": relax_events[0],
            "pace_recalibrate_count": recalibrate_count[0],
            "pace_final_max_pos_sum": config.max_pos_sum,
            "pace_freq_min_used": min(f[1] for f in freq_history) if freq_history else 0,
            "pace_freq_max_used": max(f[1] for f in freq_history) if freq_history else 0,
        }

        if verbose:
            print(
                f"[PACE] uninstalled: steps={step_count[0]} "
                f"avg_energy_factor={avg_e_factor:.4f} avg_load={avg_load:.4f}"
            )

        controller.shutdown()
        return stats

    return uninstall
