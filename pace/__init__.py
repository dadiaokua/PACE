"""PACE: Position-Aware Control for Energy-efficient LLM serving."""

from pace.config import PACEConfig
from pace.gpu import GPUFrequencyController
from pace.vllm_plugin import install_pace
from pace.working_set import DecodeStepState, active_kv_working_set, extract_decode_state

__all__ = [
    "PACEConfig",
    "GPUFrequencyController",
    "DecodeStepState",
    "active_kv_working_set",
    "extract_decode_state",
    "install_pace",
]

__version__ = "0.1.0"
