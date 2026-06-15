"""GPU frequency control via NVML or nvidia-smi."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUFrequencyController:
    """Set discrete GPU clock levels through NVML or nvidia-smi."""

    def __init__(
        self,
        gpu_index: int = 0,
        gpu_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ):
        self.gpu_index = gpu_index
        self.gpu_indices = gpu_indices or [gpu_index]
        self.verbose = verbose
        self._handle = None
        self._pynvml_available = False
        self._can_set_clocks = False
        self._freq_levels: List[int] = []
        self._max_freq: int = 0
        self._min_freq: int = 0
        self._current_freq: int = 0
        self._mode: str = "disabled"

        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._pynvml_available = True

            mem_clks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self._handle)
            gr_clks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                self._handle, mem_clks[0]
            )
            self._freq_levels = sorted(set(gr_clks))
            self._max_freq = max(self._freq_levels)
            self._min_freq = min(self._freq_levels)
            self._current_freq = self._max_freq
            self._freq_levels = self._downsample_levels(self._freq_levels, max_levels=8)

            if self.verbose:
                name = pynvml.nvmlDeviceGetName(self._handle)
                print(f"[PACE] GPU {self.gpu_index}: {name}")
                print(f"    clock range: {self._min_freq}-{self._max_freq} MHz")
                print(f"    ladder: {self._freq_levels}")

            gpu_id_str = ",".join(str(i) for i in self.gpu_indices)
            if self._try_pynvml_lock():
                self._can_set_clocks = True
                self._mode = "pynvml"
                if self.verbose:
                    print("    DVFS backend: pynvml")
            elif self._try_nvidia_smi(gpu_id_str, use_sudo=False):
                self._can_set_clocks = True
                self._mode = "nvidia-smi"
                if self.verbose:
                    print("    DVFS backend: nvidia-smi")
            elif self._try_nvidia_smi(gpu_id_str, use_sudo=True):
                self._can_set_clocks = True
                self._mode = "sudo-smi"
                if self.verbose:
                    print("    DVFS backend: sudo nvidia-smi")
            else:
                self._mode = "record_only"
                if self.verbose:
                    print("    DVFS backend: record-only (no clock control permission)")

        except ImportError:
            if self.verbose:
                print("[PACE] pynvml unavailable; probing clocks via nvidia-smi")
            self._mode = "record_only"
            self._freq_levels, self._min_freq, self._max_freq = (
                self._fallback_query_freq_via_smi(self.gpu_index)
            )
        except Exception as exc:
            if self.verbose:
                print(f"[PACE] NVML init failed: {exc}")
            self._mode = "record_only"
            self._freq_levels, self._min_freq, self._max_freq = (
                self._fallback_query_freq_via_smi(self.gpu_index)
            )

    @staticmethod
    def _downsample_levels(levels: List[int], max_levels: int = 8) -> List[int]:
        if len(levels) <= max_levels:
            return levels
        step = len(levels) // max_levels
        sampled = [levels[i * step] for i in range(max_levels)]
        if levels[-1] not in sampled:
            sampled.append(levels[-1])
        return sorted(set(sampled))

    def _try_pynvml_lock(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlDeviceSetGpuLockedClocks(
                self._handle, self._max_freq, self._max_freq
            )
            pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
            return True
        except Exception:
            return False

    def _try_nvidia_smi(self, gpu_id_str: str, use_sudo: bool) -> bool:
        base = ["sudo", "-n", "nvidia-smi"] if use_sudo else ["nvidia-smi"]
        try:
            result = subprocess.run(
                base + ["-i", gpu_id_str, "-lgc", f"{self._max_freq},{self._max_freq}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            subprocess.run(
                base + ["-i", gpu_id_str, "-rgc"],
                capture_output=True,
                timeout=10,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _fallback_query_freq_via_smi(gpu_index: int) -> Tuple[List[int], int, int]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-supported-clocks=gr",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            all_clocks = sorted({int(x.strip()) for x in out.splitlines() if x.strip()})
            if len(all_clocks) >= 2:
                levels = GPUFrequencyController._downsample_levels(all_clocks)
                return levels, all_clocks[0], all_clocks[-1]
        except Exception:
            pass
        return [1500], 1500, 1500

    def _smi_cmd(self, *args: str) -> List[str]:
        base = ["sudo", "-n", "nvidia-smi"] if self._mode == "sudo-smi" else ["nvidia-smi"]
        return base + list(args)

    def set_frequency(self, target_freq: int) -> bool:
        if target_freq == self._current_freq:
            return self._mode != "record_only"

        self._current_freq = target_freq
        gpu_id_str = ",".join(str(i) for i in self.gpu_indices)

        if self._mode == "pynvml":
            try:
                import pynvml

                pynvml.nvmlDeviceSetGpuLockedClocks(
                    self._handle, target_freq, target_freq
                )
                return True
            except Exception as exc:
                logger.debug("setGpuLockedClocks failed: %s", exc)
                return False

        if self._mode in ("nvidia-smi", "sudo-smi"):
            try:
                result = subprocess.run(
                    self._smi_cmd("-i", gpu_id_str, "-lgc", f"{target_freq},{target_freq}"),
                    capture_output=True,
                    timeout=10,
                )
                return result.returncode == 0
            except Exception:
                return False

        return False

    def reset_frequency(self) -> None:
        gpu_id_str = ",".join(str(i) for i in self.gpu_indices)
        if self._mode == "pynvml":
            try:
                import pynvml

                pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
            except Exception:
                pass
        elif self._mode in ("nvidia-smi", "sudo-smi"):
            try:
                subprocess.run(
                    self._smi_cmd("-i", gpu_id_str, "-rgc"),
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        self._current_freq = self._max_freq

    def shutdown(self) -> None:
        self.reset_frequency()
        if self._pynvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def freq_levels(self) -> List[int]:
        return self._freq_levels

    @property
    def max_freq(self) -> int:
        return self._max_freq

    @property
    def min_freq(self) -> int:
        return self._min_freq

    @property
    def current_freq(self) -> int:
        return self._current_freq

    @property
    def is_real_dvfs(self) -> bool:
        return self._mode in ("pynvml", "nvidia-smi", "sudo-smi")

    @property
    def mode(self) -> str:
        return self._mode
