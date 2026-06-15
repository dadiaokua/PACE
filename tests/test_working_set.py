from pace.config import PACEConfig
from pace.working_set import active_kv_working_set


def test_active_kv_working_set():
    assert active_kv_working_set([2, 4, 12, 8]) == 26.0


def test_compute_load():
    cfg = PACEConfig(max_batch=4, max_pos_sum=40.0, fixed_cost_weight=0.35)
    load = cfg.compute_load(batch_size=4, position_sum=26.0)
    assert 0.0 < load <= 1.0
    assert load > cfg.compute_load(batch_size=4, position_sum=8.0)
