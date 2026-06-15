"""PACE controller configuration and load model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PACEConfig:
    """Feedforward DVFS policy driven by the working-set energy law."""

    # E_step = C + k * S_KV  =>  alpha ~= C / (C + k * S_ref)
    fixed_cost_weight: float = 0.35
    slo_margin: float = 1.2

    max_position: float = 12000.0
    max_batch: int = 64
    max_pos_sum: float = 0.0

    throughput_guard: float = 0.90
    tighten_factor: float = 0.90
    relax_factor: float = 1.02
    min_slo_ratio: float = 0.8
    max_slo_ratio: float = 1.5

    warmup_steps: int = 500
    window_size: int = 100
    auto_calibrate: bool = True
    recalibrate_interval: int = 200
    recalibrate_window: int = 500

    prefill_boost: bool = True
    min_freq_ratio: float = 0.10
    freq_switch_cooldown: int = 2

    static_power_frac: float = 0.30
    power_exponent: float = 1.5

    def compute_load(self, batch_size: int, position_sum: float) -> float:
        if batch_size == 0:
            return 0.0

        alpha = self.fixed_cost_weight
        batch_ratio = min(1.0, batch_size / max(1, self.max_batch))
        pos_denom = (
            self.max_pos_sum
            if self.max_pos_sum > 0
            else self.max_batch * self.max_position
        )
        pos_ratio = min(1.0, position_sum / max(1.0, pos_denom))
        return alpha * batch_ratio + (1.0 - alpha) * pos_ratio

    def select_frequency(
        self,
        batch_size: int,
        position_sum: float,
        freq_levels: List[int],
        slo_margin_override: Optional[float] = None,
    ) -> int:
        if batch_size == 0:
            return freq_levels[0]

        load = self.compute_load(batch_size, position_sum)
        margin = slo_margin_override if slo_margin_override is not None else self.slo_margin
        min_freq_ratio = max(self.min_freq_ratio, min(1.0, load / margin))

        f_min = freq_levels[0]
        f_max = freq_levels[-1]
        target_freq = f_min + min_freq_ratio * (f_max - f_min)

        for freq in freq_levels:
            if freq >= target_freq:
                return freq
        return freq_levels[-1]

    def energy_factor(self, freq: int, max_freq: int) -> float:
        ratio = freq / max(1, max_freq)
        return self.static_power_frac + (1 - self.static_power_frac) * (
            ratio ** self.power_exponent
        )
