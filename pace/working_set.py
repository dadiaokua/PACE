"""Active KV working set helpers for continuous-batching schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class DecodeStepState:
    batch_size: int
    position_sum: float
    avg_position: float
    has_prefill: bool
    positions: List[int]


def active_kv_working_set(positions: Sequence[int]) -> float:
    """Return S_KV(t) = sum_i p_i for the active batch."""
    return float(sum(positions))


def extract_decode_state(running_queue: Iterable) -> DecodeStepState:
    """Read batch size and token positions from a vLLM running queue."""
    positions: List[int] = []
    has_prefill = False

    for seq_group in running_queue:
        seqs = seq_group.get_seqs()
        if not seqs:
            continue
        out_len = seqs[0].get_output_len()
        positions.append(out_len)
        if out_len == 0:
            has_prefill = True

    batch_size = len(positions)
    position_sum = active_kv_working_set(positions)
    avg_position = position_sum / batch_size if batch_size else 0.0

    return DecodeStepState(
        batch_size=batch_size,
        position_sum=position_sum,
        avg_position=avg_position,
        has_prefill=has_prefill,
        positions=positions,
    )
