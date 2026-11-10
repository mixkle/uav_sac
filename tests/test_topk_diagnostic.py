import copy
import itertools

import numpy as np
import pytest
import torch

from uav_research.topk_diagnostic import (
    METRICS, assignment_metrics, audit_transition, collect, maximum_assignment,
    score_array, topk_and_fill, validate_assignment,
)
from uav_research import kernels
from uav_research.algorithms import make_learner
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.runtime import configure_runtime, isolated_rng, seed_all


def brute_count(edges, capacity):
    best = 0
    u, m = edges.shape
    for assignment in itertools.product(range(-1, u), repeat=m):
        own = np.array(assignment)
        used = np.flatnonzero(own >= 0)
        if (edges[own[used], used].all()
                and np.bincount(own[used], minlength=u).max(initial=0) <= capacity):
            best = max(best, len(used))
    return best


@pytest.mark.parametrize('capacity', [1, 2, 4])
def test_exact_maximum_against_exhaustive_small_graphs(capacity):
    for bits in itertools.product([False, True], repeat=6):
        edges = np.array(bits).reshape(2, 3)
        result = maximum_assignment(edges, capacity)
        validate_assignment(result, edges, capacity)
        assert (result >= 0).sum() == brute_count(edges, capacity)
        np.testing.assert_array_equal(result, maximum_assignment(edges, capacity))
    rng = np.random.default_rng(81)
    for _ in range(15):
        edges = rng.random((3, 5)) < .5
        result = maximum_assignment(edges, capacity)
        assert (result >= 0).sum() == brute_count(edges, capacity)


def test_duplicate_proposals_and_fill():
    e = np.ones((2, 2), bool)
    scores = np.array([[.9, .8], [.9, .7]])
    p0, p1, proposal, valid, extra = topk_and_fill(e, e, scores, 1)
    np.testing.assert_array_equal(p0, [0, -1])
    np.testing.assert_array_equal(p1, [0, 1])
    assert extra.tolist() == [1, 0, 1, 1, 1, 0]
    assert (p0 >= 0).sum() == valid.any(0).sum()


def test_invalidated_proposal_uses_previously_scored_candidate():
    before = np.ones((1, 2), bool)
    after = np.array([[False, True]])
    p0, p1, _, _, extra = topk_and_fill(before, after, np.array([[.9, .8]]), 1)
    assert (p0 >= 0).sum() == 0
    np.testing.assert_array_equal(p1, [-1, 0])
    assert extra[1] == 1 and extra[4] == 1
    # An edge first appearing after motion must not be used.
    p0, p1, *_ = topk_and_fill(np.array([[True, False]]), after, np.ones((1, 2)), 1)
    assert not (p1 >= 0).any()


def test_augmenting_path_required_and_committed_requests_counted():
    e = np.array([[True, True], [True, False]])
    scores = np.array([[.9, .7], [.8, 0.]])
    p0, p1, *_ = topk_and_fill(e, e, scores, 1)
    np.testing.assert_array_equal(p0, [0, -1])
    np.testing.assert_array_equal(p1, p0)
    assert (maximum_assignment(e, 1) >= 0).sum() == 2
    # UAV B initially chooses UE0 too; here an additional request to an already
    # committed UE is counted, not skipped using a global free-UE list.
    e = np.ones((3, 3), bool)
    scores = np.array([[.95, .1, .1], [.9, .8, .1], [.85, .7, .6]])
    p0, p1, _, _, extra = topk_and_fill(e, e, scores, 1)
    assert (p1 >= 0).sum() == 3
    assert extra[3] == 2 and extra[4] == 3 and extra[5] == 1


def test_empty_capacity_and_ties():
    for e in (np.zeros((2, 3), bool), np.ones((2, 3), bool)):
        p0, p1, proposal, *_ = topk_and_fill(e, e, np.full(e.shape, .5), 2)
        validate_assignment(p0, e, 2)
        validate_assignment(p1, e, 2)
        if e.any():
            np.testing.assert_array_equal(p0, [0, 0, -1])
            np.testing.assert_array_equal(p1, [0, 0, 1])
            assert proposal[:, :2].all() and not proposal[:, 2:].any()
        else:
            assert not (p1 >= 0).any()
    with pytest.raises(ValueError):
        topk_and_fill(e, e, np.full(e.shape, np.nan), 2)


def test_random_scores_are_keyed_not_execution_order():
    e = np.array([[True, False, True], [True, True, False]])
    first = score_array('random', e, np.zeros(e.shape), -3, 10000, 17, 4)
    score_array('random', e, np.zeros(e.shape), -3, 10001, 27, 8)
    np.testing.assert_array_equal(first, score_array('random', e, e, -3, 10000, 17, 4))
    assert not np.array_equal(first, score_array('random', e, e, -3, 10000, 17, 5))
    assert ((first[e] > 0) & (first[e] < 1)).all()
    assert not first[~e].any()


def small_cfg():
    c = load_config('configs/mappo_coverage_shared_1m.yaml')
    c.env.num_uavs, c.env.num_users = 3, 9
    c.env.horizon = 5
    c.run.cpu_threads, c.run.numba_threads, c.run.num_envs = 1, 1, 2
    c.algo.hidden = 16
    return c


def test_reward_failure_and_deficit_invariants():
    cfg = small_cfg()
    configure_runtime(cfg)
    env = UAVEnvironment(cfg.env)
    env.reset(52)
    before = (env.sensed & (env.Y >= cfg.env.sinr_threshold_db)
              & (env.B.min(-1)[:, None] <= 3))
    y = env.Y.copy()
    env.step(np.zeros((3, 2), np.float32))
    after = (env.sensed & (env.Y >= cfg.env.sinr_threshold_db)
             & (env.B.min(-1)[:, None] <= 3))
    scores = score_array('teacher', before, y, -3, 52, 0)
    owners, _, _, rows = audit_transition(before, after, scores, 20, env.rates,
        env.post_hops.min(-1) <= 3, env.B.min(-1) <= 3, cfg.env.rate_reference_bps)
    assert rows.shape == (2, len(METRICS))
    for own, row in zip(owners, rows):
        counts = np.bincount(own[own >= 0], minlength=3)
        retained, _, _ = kernels.failure_coverage(env.A, env.direct, counts, env.B, 9)
        assert row[METRICS.index('robustness')] == pytest.approx(retained.mean(), abs=1e-14)
        assert row[6:10].sum() == pytest.approx(row[10], abs=1e-14)
        assert 9-row[0] == row[11]+row[14]+row[15]
    assert rows[0, 5] == rows[1, 5]


def test_shadow_diagnostics_do_not_change_actual_trajectory_or_rng(tmp_path):
    cfg = small_cfg()
    configure_runtime(cfg)
    with isolated_rng():
        seed_all(7)
        policy = make_learner(cfg, 'cpu')
    policy.actor.eval().requires_grad_(False)
    before_parameters = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    def observe(scene, t, before, sinr, after, env):
        for repeat in range(3):
            scores = score_array('random', before, sinr, -3, scene, t, repeat)
            audit_transition(before, after, scores, cfg.env.capacity, env.rates,
                             env.post_hops.min(-1) <= cfg.env.num_uavs,
                             env.B.min(-1) <= cfg.env.num_uavs, cfg.env.rate_reference_bps)
    mute = lambda *args, **kwargs: None
    collect(cfg, policy, tmp_path/'off', [10000, 10001], 5, progress=mute)
    collect(cfg, policy, tmp_path/'on', [10000, 10001], 5, progress=mute, observer=observe)
    for seed in (10000, 10001):
        with np.load(tmp_path/'off'/'trajectories'/f'scene_{seed}.npz') as a, np.load(
                tmp_path/'on'/'trajectories'/f'scene_{seed}.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for k, v in policy.actor.state_dict().items():
        torch.testing.assert_close(v, before_parameters[k], rtol=0, atol=0)
    assert all(p.grad is None for p in policy.actor.parameters())


def test_saved_audit_resume_and_report_rebuild_without_policy(tmp_path):
    import json
    from uav_research.runtime import atomic_json
    from uav_research.topk_diagnostic import audit_scene, file_hash
    from uav_research.topk_report import build_report
    cfg = small_cfg(); configure_runtime(cfg)
    with isolated_rng():
        seed_all(11)
        policy = make_learner(cfg, 'cpu')
    policy.actor.eval().requires_grad_(False)
    collect(cfg, policy, tmp_path, [10000,10001], 5, progress=lambda *a,**k:None)
    for seed in (10000,10001):
        source=tmp_path/'trajectories'/f'scene_{seed}.npz'
        audit_scene(source,tmp_path/'audit'/f'scene_{seed}.npz',cfg,repeats=3)
        with np.load(tmp_path/'audit'/f'scene_{seed}.npz') as d:
            assert str(d['trajectory_sha256'])==file_hash(source)
            assert d['metrics'].shape==(5,5,2,len(METRICS))
            assert np.isfinite(d['metrics']).all()
    atomic_json(tmp_path/'provenance.json',dict(
        source_config=cfg.as_dict(),checkpoint='frozen-test-only',checkpoint_env_steps=0,
        settings=dict(scenes=[10000,10001],random_repeats=3)))
    first=build_report(tmp_path)
    saved_summary=(tmp_path/'summary.json').read_bytes()
    with np.load(tmp_path/'figure_data.npz') as d:
        arrays={k:d[k].copy() for k in d.files}
    # No policy argument or RNG/model access is needed to rebuild.
    second=build_report(tmp_path)
    assert first==second
    assert saved_summary==(tmp_path/'summary.json').read_bytes()
    with np.load(tmp_path/'figure_data.npz') as d:
        for key,value in arrays.items():np.testing.assert_array_equal(value,d[key])
    assert (tmp_path/'report.md').exists()
    assert not first['bootstrap']['training_seed_interval']


def test_failed_atomic_scene_save_preserves_previous_data(tmp_path,monkeypatch):
    from uav_research import topk_diagnostic as diagnostic
    path=tmp_path/'scene.npz'
    diagnostic.atomic_npz(path,value=np.array([1]))
    before=path.read_bytes()
    def reject(*args):raise OSError('simulated failed publish')
    monkeypatch.setattr(diagnostic.os,'replace',reject)
    with pytest.raises(OSError):diagnostic.atomic_npz(path,value=np.array([2]))
    assert path.read_bytes()==before
    assert not path.with_suffix('.npz.tmp').exists()


def test_capacity_counterfactual_preserves_radio_graph_and_rescales_saved_rates(tmp_path):
    from uav_research.environment import physical_links
    from uav_research.topk_diagnostic import audit_scene, file_hash
    source=small_cfg();configure_runtime(source)
    target=copy.deepcopy(source);target.env.capacity=10
    rng=np.random.default_rng(93)
    xy=rng.uniform(0,3500,(3,2));ue=rng.uniform(500,3000,(9,2))
    a=physical_links(source.env,xy,ue);b=physical_links(target.env,xy,ue)
    np.testing.assert_allclose(a[2],b[2],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(b[3],2*a[3],atol=1e-9,rtol=1e-12)
    for k in (0,1,4,5):np.testing.assert_array_equal(a[k],b[k])
    with isolated_rng():
        seed_all(19);policy=make_learner(source,'cpu')
    collect(source,policy,tmp_path,[10000,10001],5,progress=lambda *a,**k:None)
    p=tmp_path/'trajectories'/'scene_10000.npz'
    before=file_hash(p)
    audit_scene(p,tmp_path/'original.npz',source,repeats=2)
    audit_scene(p,tmp_path/'changed.npz',target,repeats=2,trajectory_capacity=20)
    assert file_hash(p)==before
    with np.load(tmp_path/'original.npz') as a,np.load(tmp_path/'changed.npz') as b:
        assert int(b['audit_capacity'])==10 and int(b['trajectory_capacity'])==20
        assert float(b['link_rate_scale'])==2
        # Only 9 UEs exist: neither N=10 nor N=20 binds in this case.
        np.testing.assert_array_equal(a['owners'],b['owners'])
        np.testing.assert_array_equal(a['teacher_scores'],b['teacher_scores'])
        for metric in ('coverage','robustness','disconnection'):
            j=METRICS.index(metric)
            np.testing.assert_array_equal(a['metrics'][...,j],b['metrics'][...,j])
        for metric in ('tau','throughput_bps','reward_throughput'):
            j=METRICS.index(metric)
            np.testing.assert_allclose(b['metrics'][...,j],2*a['metrics'][...,j],rtol=1e-13)


def test_reused_trajectory_report_keeps_source_and_audit_capacity_separate(tmp_path):
    from uav_research.runtime import atomic_json
    from uav_research.topk_diagnostic import audit_scene
    from uav_research.topk_report import build_report
    cfg=small_cfg();configure_runtime(cfg)
    with isolated_rng():seed_all(21);policy=make_learner(cfg,'cpu')
    ref=tmp_path/'reference';out=tmp_path/'variant';out.mkdir()
    collect(cfg,policy,ref,[10000,10001],5,progress=lambda *a,**k:None)
    parent=dict(source_config=cfg.as_dict(),checkpoint='test-frozen-only',checkpoint_env_steps=0,
                settings=dict(scenes=[10000,10001],random_repeats=2))
    atomic_json(ref/'provenance.json',parent)
    for scene in (10000,10001):
        audit_scene(ref/'trajectories'/f'scene_{scene}.npz',ref/'audit'/f'scene_{scene}.npz',cfg,2)
    build_report(ref)
    target=copy.deepcopy(cfg);target.env.capacity=10
    atomic_json(out/'provenance.json',dict(parent,audit_config=target.as_dict(),
                trajectory_source=str(ref),comparison_reference=str(ref)))
    for scene in (10000,10001):
        audit_scene(ref/'trajectories'/f'scene_{scene}.npz',out/'audit'/f'scene_{scene}.npz',target,2,
                    trajectory_capacity=20)
    summary=build_report(out)
    assert not (out/'trajectories').exists()
    assert summary['audit_capacity']==10 and summary['trajectory_capacity']==20
    assert summary['link_rate_scale']==2
    assert (out/'comparison_capacity20.md').exists()
    assert '容量10' in (out/'comparison_capacity20.md').read_text()
    assert '容量 **10 UE/UAV**' in (out/'report.md').read_text()


def test_two_term_reward_changes_no_association_and_matches_saved_contributions(tmp_path):
    from uav_research.runtime import atomic_json
    from uav_research.topk_diagnostic import audit_scene
    from uav_research.topk_report import build_report
    cfg=small_cfg();configure_runtime(cfg)
    cfg.env.capacity=10;cfg.env.throughput_weight=.1
    with isolated_rng():seed_all(25);policy=make_learner(cfg,'cpu')
    collect(cfg,policy,tmp_path,[10000,10001],5,progress=lambda *a,**k:None)
    weights=[1.,.1,0.,0.]
    for scene in (10000,10001):
        path=tmp_path/'trajectories'/f'scene_{scene}.npz'
        audit_scene(path,tmp_path/f'old_{scene}.npz',cfg,2)
        audit_scene(path,tmp_path/'audit'/f'scene_{scene}.npz',cfg,2,reward_weights=weights)
        with np.load(tmp_path/f'old_{scene}.npz') as a,np.load(tmp_path/'audit'/f'scene_{scene}.npz') as b:
            np.testing.assert_array_equal(a['owners'],b['owners'])
            np.testing.assert_array_equal(a['metrics'][...,:6],b['metrics'][...,:6])
            np.testing.assert_array_equal(b['reward_weights'],weights)
            np.testing.assert_array_equal(b['metrics'][...,8:10],0)
            np.testing.assert_allclose(b['metrics'][...,10],b['metrics'][...,1]+.1*b['metrics'][...,3],atol=1e-14)
    atomic_json(tmp_path/'provenance.json',dict(source_config=cfg.as_dict(),
        checkpoint='test-only',checkpoint_env_steps=123,source_run_env_steps=1000000,
        audit_reward_weights=weights,settings=dict(scenes=[10000,10001],random_repeats=2)))
    result=build_report(tmp_path)
    assert result['audit_reward_weights']==weights
    report=(tmp_path/'report.md').read_text()
    assert '训练奖励为 **覆盖+0.1×归一化数据率**' in report
    assert '采用纯覆盖奖励' not in report
    for key in ('reward_robustness','reward_disconnection'):
        assert result['groups']['teacher']['P0']['all'][key]['mean']==0


def test_source_validation_uses_logged_coverage_not_return_divided_by_horizon(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('topk_entry',Path('scripts/diagnose_topk_capacity.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=load_config('configs/mappo_coverage_throughput_capacity10_1m.yaml')
    assert module.validate_environment(cfg,cfg)=={}
    wrong=copy.deepcopy(cfg);wrong.env.capacity=20
    with pytest.raises(ValueError,match='capacity'):module.validate_environment(wrong,cfg)
    seeds=list(range(10000,10032));folder=tmp_path/'trajectories';folder.mkdir()
    for seed in seeds:
        np.savez_compressed(folder/f'scene_{seed}.npz',scene_seed=seed,
            actions=np.zeros((400,1,2)),source_metrics=np.tile([.25,.3,0,0],(400,1)))
    p=dict(settings=dict(scenes=seeds,steps=400),expected_source_coverage=.25,expected_source_return=120.)
    result=module.check_source_coverage(tmp_path,p)
    assert result['passed'] and result['coverage']==.25
    assert result['episode_return']==pytest.approx(120.)
    p['expected_source_return']=100.
    with pytest.raises(AssertionError):module.check_source_coverage(tmp_path,p)
