import numpy as np
import pytest
from uav_research.post_deployment_metrics import trajectory_metrics, summarize, load_trajectory


def test_exact_threshold_and_window_entry_transition():
    c = np.full((1, 400), .8)
    c[:, 300:] = .7  # transition 300 -> 301 belongs to last100
    c[:, 350:] = .6  # transition 350 -> 351 belongs to last50
    m = trajectory_metrics(c, [10000])
    assert m['tail100_drop_count'][0] == 2
    assert m['tail50_drop_count'][0] == 1
    assert m['tail100_max_drop_pp'][0] == pytest.approx(10)
    assert m['tail50_coverage_pct'][0] == pytest.approx(60)
    assert m['tail100_coverage_pct'][0] == pytest.approx(65)


def test_mean_temporal_sd_is_not_rms_or_between_scene_sd():
    c = np.full((2, 400), .2)
    c[1, :] = np.tile([.4, .8], 200)
    r, _, _ = summarize([trajectory_metrics(c, [1, 2])])
    assert r['tail100_coverage_std_pp'] == pytest.approx(10)
    assert r['tail50_coverage_std_pp'] == pytest.approx(10)
    constant = trajectory_metrics(np.array([[.2]*400, [.8]*400]), [1, 2])
    assert summarize([constant])[0]['tail100_coverage_std_pp'] == pytest.approx(0)


def test_repeat_statistics_before_average_prevents_noise_cancellation():
    c = np.tile([.4, .8], 200)[None, :]
    r, _, _ = summarize([trajectory_metrics(c, [1]), trajectory_metrics(1.2-c, [1])])
    assert r['tail100_coverage_std_pp'] == pytest.approx(20)
    assert r['action_repetitions'] == 2


def test_per_scene_quantile_not_pooled_quantile():
    c = np.array([[.2]*400, [.8]*400])
    r, _, _ = summarize([trajectory_metrics(c, [1, 2])])
    assert r['tail100_coverage_p05_pct'] == pytest.approx(50)
    assert np.quantile(c[:, -100:], .05) == pytest.approx(.2)


def test_return_connectivity_and_physical_throughput():
    c = np.full((2, 400), .5)
    reward = np.full_like(c, -.3)
    m = trajectory_metrics(c, [1, 2], throughput=np.full_like(c, 123e6),
                           rewards=reward, disconnection=np.full_like(c, .25))
    r, _, _ = summarize([m])
    assert r['episode_return'] == pytest.approx(-120)
    assert r['tail100_connected_pct'] == pytest.approx(75)
    assert r['final_throughput_mbps'] == pytest.approx(123)


def test_formation_first_crossing_without_claim_of_sustained_deployment():
    c = np.zeros((3, 400))
    c[0, 20] = 1
    c[0, 300:] = .5
    c[1] = .7
    m = trajectory_metrics(c, [1, 2, 3])
    np.testing.assert_array_equal(m['formation_step'], [21, 1, 1])


def test_missing_optional_evidence_remains_blank(tmp_path):
    p = tmp_path/'minimal.npz'
    np.savez(p, metrics=np.full((2, 400, 1), .5), names=['coverage'], scene_seeds=[1, 2])
    m = load_trajectory(p)
    r, ci, _ = summarize([m])
    for key in ['episode_return', 'tail100_throughput_mbps', 'final_connected_pct']:
        assert r[key] is None and ci[key] is None


def test_legacy_tau_and_explicit_reward_override(tmp_path):
    p = tmp_path/'shadow.npz'
    a = np.zeros((1, 400, 4))
    a[:] = [.4, .5, 1, 2]
    np.savez(p, metrics=a, names=['coverage', 'tau', 'reward', 'sinr_score_reward'], seeds=[9])
    r = load_trajectory(p, throughput_scale_bps=750e6, reward_name='sinr_score_reward')
    assert r['tail50_throughput_mbps'][0] == pytest.approx(375)
    assert r['episode_return'][0] == pytest.approx(800)


def test_scene_alignment_and_invalid_inputs():
    c = np.full((2, 400), .5)
    with pytest.raises(AssertionError):
        summarize([trajectory_metrics(c, [1, 2]), trajectory_metrics(c, [2, 1])])
    with pytest.raises(ValueError):
        trajectory_metrics(c, [1, 1])
    with pytest.raises(ValueError):
        trajectory_metrics(c, [1, 2], rewards=np.zeros((2, 399)))


def test_loader_rejects_partial_episode(tmp_path):
    p = tmp_path/'partial.npz'
    np.savez(p, metrics=np.full((1, 399, 1), .5), names=['coverage'], scene_seeds=[1])
    with pytest.raises(ValueError, match='complete 400-step'):
        load_trajectory(p)
