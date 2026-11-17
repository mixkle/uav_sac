import numpy as np
import pytest

from uav_research.system_ablation.report import episode_metrics
from uav_research.post_deployment_metrics import trajectory_metrics


def test_late_ablation_metrics_match_shared_protocol():
    rng = np.random.default_rng(73)
    c = rng.integers(0,151,(2,400))/150.
    rate = rng.uniform(1e6,5e8,c.shape)
    disconnected = rng.integers(0,19,c.shape)/18.
    rewards = c+.1*rate/750e6
    data = np.stack([c,rate,rewards,disconnected],axis=-1)
    rows = episode_metrics(data,['coverage','throughput_bps','reward','disconnection_ratio'],[10000,10001])
    reference = trajectory_metrics(c,[10000,10001],throughput=rate,rewards=rewards,disconnection=disconnected)
    for i,row in enumerate(rows):
        for n in (100,50):
            assert row[f'last{n}_coverage']*100 == pytest.approx(reference[f'tail{n}_coverage_pct'][i])
            assert row[f'last{n}_coverage_std_pp'] == pytest.approx(reference[f'tail{n}_coverage_std_pp'][i])
            assert row[f'last{n}_throughput_mbps'] == pytest.approx(reference[f'tail{n}_throughput_mbps'][i])
            assert row[f'last{n}_drop_events_10pp'] == reference[f'tail{n}_drop_count'][i]
            assert row[f'last{n}_max_drop_pp'] == pytest.approx(reference[f'tail{n}_max_drop_pp'][i])
        assert row['final_connected_fraction']*100 == pytest.approx(reference['final_connected_pct'][i])
        assert row['formation_step'] == reference['formation_step'][i]
        assert row['maintain_coverage'] == pytest.approx(c[i,-100:].mean()-c[i,50:150].mean())
