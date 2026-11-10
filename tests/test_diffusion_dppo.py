"""Contracts that can invalidate scientific conclusions: probability, timing,
isolation, PopArt/GAE, and exact interrupted CPU continuation."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from uav_research.config import load_config, config_from_dict, Config
from uav_research.runtime import seed_all, atomic_save, rng_state
from uav_research.algorithms import make_learner, mappo_value_loss, gae
from uav_research.environment import UAVEnvironment
from uav_research.topk_diagnostic import _topk_and_fill
from uav_research.diffusion_dppo.config import baseline_differences, contract
from uav_research.diffusion_dppo.observations import (ServiceObservation, offsets, DiffusionChain,
    ServiceAction, stack, stack_chains)
from uav_research.diffusion_dppo.environment import ServiceEnvironment, ServiceVector, resolve_topk
from uav_research.diffusion_dppo.policy import DiffusionActor
from uav_research.diffusion_dppo.learner import DiffusionLearner, DppoRollout
from uav_research.diffusion_dppo.storage import load, artifact, atomic_directory


@pytest.fixture
def cfg():
    c = load_config("configs/diffusion_dppo_local_smoke.yaml")
    torch.set_num_threads(2)
    c.diffusion.minimum_free_gib = 0
    return c


def synthetic(cfg, lengths=(0, 3, 2, 4), candidates=(0, 2, 0, 4)):
    rng = np.random.default_rng(73)
    features = rng.normal(size=(sum(lengths), 5)).astype(np.float32)
    mask = np.concatenate([np.arange(n) < c for n, c in zip(lengths, candidates)])
    return ServiceObservation(rng.normal(size=(len(lengths), cfg.env.fixed_obs_dim)).astype(np.float32),
        features, offsets(lengths), np.concatenate([np.arange(n) for n in lengths]), mask)


def nested_equal(a, b, atol=0):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=atol, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, atol=atol, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            nested_equal(a[k], b[k], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            nested_equal(x, y, atol)
    else:
        assert a == b


def test_configs_and_isolation(cfg):
    base = load_config("configs/mappo_coverage_throughput_capacity10_1m.yaml")
    main = load_config("configs/diffusion_dppo_capacity10_coverage_throughput_1m.yaml")
    assert baseline_differences(main, base) == {}
    assert main.env.state_dim == 6292 and main.env.fixed_obs_dim == 105
    assert main.algo.ppo_clip == .1 and main.diffusion.value_clip == .2
    assert type(config_from_dict(base.as_dict())) is Config
    for field, value in (("entropy_coef", .01), ("actor_sharing", "independent")):
        bad = copy.deepcopy(cfg)
        setattr(bad.algo, field, value)
        with pytest.raises(ValueError):
            bad.validate()
    bad = copy.deepcopy(cfg)
    bad.diffusion.fit_seeds = bad.run.validation_seeds
    with pytest.raises(ValueError, match="overlap"):
        bad.validate()


def test_continuation_runtime_preserves_historical_contract(cfg):
    historical = cfg.as_dict()
    historical['diffusion'].pop('actor_update_devices')
    from uav_research.diffusion_dppo.config import from_dict
    old_contract = contract(from_dict(historical))
    extended = copy.deepcopy(cfg)
    extended.run.total_steps = 3_000_000
    extended.run.device = 'cuda:0'
    extended.diffusion.actor_update_devices = ['cuda:0', 'cuda:1']
    extended.diffusion.microbatch_agents = 16384
    extended.validate()
    assert contract(extended) == old_contract
    for devices in (['cuda:0', 'cuda:0'], ['cpu'], 'cuda:0', ['cuda:1']):
        extended.diffusion.actor_update_devices = devices
        with pytest.raises(ValueError, match='device'):
            extended.validate()


def test_fork_continuation_preserves_source_and_history(cfg, tmp_path):
    from uav_research.diffusion_dppo.continuation import prepare_continuation
    from uav_research.diffusion_dppo.storage import save_online_full
    source, output = tmp_path/'source', tmp_path/'continued'
    source.mkdir()
    atomic_save(source/'step0.pt', {'preserve': True})
    for name in ('metrics.csv', 'validation.csv'):
        (source/name).write_text('env_steps,return\n0,1\n8,2\n16,3\n')
    full = artifact(cfg, 'full', env_steps=8, learner={'updates': 1})
    checkpoint = save_online_full(source, cfg, full)
    before = {str(f.relative_to(source)): f.read_bytes() for f in source.rglob('*') if f.is_file()}
    result = prepare_continuation(cfg, checkpoint, output)
    assert result['source_steps'] == 8 and result['additional_steps'] == 8
    assert (output/'metrics.csv').read_text() == 'env_steps,return\n0,1\n8,2\n'
    assert (output/'step0.pt').read_bytes() == (source/'step0.pt').read_bytes()
    assert prepare_continuation(cfg, checkpoint, output) == result
    assert before == {str(f.relative_to(source)): f.read_bytes() for f in source.rglob('*') if f.is_file()}
    with pytest.raises(ValueError, match='preserve'):
        prepare_continuation(cfg, checkpoint, source)
    # An already-used output is not overwritten, even if its marker is absent.
    other = tmp_path/'other'
    other.mkdir()
    with pytest.raises(FileExistsError):
        prepare_continuation(cfg, checkpoint, other)


def test_replica_rng_portable_to_single_gpu(monkeypatch):
    from uav_research.diffusion_dppo.storage import restore_online_rng
    import uav_research.runtime as runtime
    received = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(runtime, 'restore_rng', received.append)
    data = {'cuda': ['primary', 'unused_replica'], 'torch': 'cpu_state'}
    restore_online_rng(data)
    assert received == [{'cuda': ['primary'], 'torch': 'cpu_state'}]
    assert data['cuda'] == ['primary', 'unused_replica']


def test_dual_artifact_can_be_evaluated_on_cpu(cfg, tmp_path):
    from uav_research.diffusion_dppo.report import evaluate_checkpoint
    learner = DiffusionLearner(cfg, 'cpu')
    dual_cfg = copy.deepcopy(cfg)
    dual_cfg.run.device = 'cuda:0'
    dual_cfg.diffusion.actor_update_devices = ['cuda:0', 'cuda:1']
    dual_cfg.validate()
    checkpoint = tmp_path/'dual.pt'
    atomic_save(checkpoint, artifact(dual_cfg, 'model', env_steps=8, learner=learner.state_dict()))
    result = evaluate_checkpoint(checkpoint, tmp_path/'evaluated', device='cpu',
                                 seeds=[10000], moment_repeats=0)
    assert np.isfinite(result['summary']['return'])


@pytest.mark.parametrize("capacity", [1, 2, 10])
def test_protocol_matches_independent_p0(capacity):
    rng = np.random.default_rng(24)
    for _ in range(30):
        before, after = rng.random((4, 9))>.3, rng.random((4, 9))>.2
        scores = rng.choice([.2, .5, .8], (4, 9))
        c, proposal, valid = resolve_topk(before, after, scores, capacity)
        reference, _, p, v, _ = _topk_and_fill(before, after, scores, capacity)
        np.testing.assert_array_equal(c.argmax(0)[c.any(0)], reference[reference >= 0])
        np.testing.assert_array_equal(c.any(0), reference >= 0)
        np.testing.assert_array_equal(proposal, p)
        np.testing.assert_array_equal(valid, v)
        assert np.all(c.sum(0) <= 1) and np.all(c.sum(1) <= capacity)
        assert np.all(~c.astype(bool) | (before & after))


def test_topk_invalid_no_refill_and_new_discovery():
    before = np.array([[1, 1, 0], [0, 1, 0]], bool)
    after = np.array([[0, 1, 1], [0, 1, 1]], bool)
    c, p, _ = resolve_topk(before, after, np.array([[.9, .8, 1], [0, .7, 1]]), 1)
    assert p[0, 0] and not c[0].any()  # no promotion to second choice
    assert c[1, 1] and not c[:, 2].any()  # first discovered after move must wait


def test_observation_routing_and_duplicate_selection(cfg):
    obs = synthetic(cfg)
    selected = obs.take([3, 0, 3, 1])
    np.testing.assert_array_equal(selected.candidate_lengths, [4, 0, 4, 2])
    np.testing.assert_array_equal(selected.fixed[[0]], selected.fixed[[2]])
    assert selected.legacy().users.shape[1] == 4
    packed = stack([obs, obs])
    assert packed.fixed.shape == (2, 4, cfg.env.fixed_obs_dim)
    np.testing.assert_array_equal(packed.take(range(4)).users, obs.users)


def test_cpu_routing_validates_unsafe_segment_inputs(cfg):
    from uav_research.diffusion_dppo.policy import mean_segments, sum_segments
    obs = synthetic(cfg)
    tensor = obs.tensor("cpu")
    np.testing.assert_array_equal(tensor.candidate_indices.numpy(), np.flatnonzero(obs.service_mask))
    values = torch.randn(len(obs.users), 3, requires_grad=True)
    lengths = tensor.lengths
    expected_sum = torch.segment_reduce(values, "sum", lengths=lengths)
    expected_mean = torch.nan_to_num(torch.segment_reduce(values, "mean", lengths=lengths), nan=0.)
    torch.testing.assert_close(sum_segments(values, lengths), expected_sum)
    torch.testing.assert_close(mean_segments(values, lengths), expected_mean)
    actual_grad, = torch.autograd.grad(sum_segments(values, lengths).sum() +
        mean_segments(values, lengths).sum(), values, retain_graph=True)
    reference_grad, = torch.autograd.grad(expected_sum.sum()+expected_mean.sum(), values)
    torch.testing.assert_close(actual_grad, reference_grad)
    for field in ("offsets", "ue_ids", "service_mask"):
        bad = copy.deepcopy(obs)
        setattr(bad, field, getattr(bad, field).astype(float))
        with pytest.raises(ValueError):
            bad.__post_init__()
    bad = copy.deepcopy(obs)
    bad.offsets = np.array([0, 4, 3, 5, 9], dtype=np.uint64)
    with pytest.raises(ValueError):
        bad.__post_init__()


def test_initialization_physics_and_restore(cfg):
    e, old = ServiceEnvironment(cfg.env), UAVEnvironment(cfg.env)
    obs, state, info = e.reset(77)
    oo, ss, oi = old.reset(77)
    np.testing.assert_array_equal(obs.fixed, oo.fixed)
    np.testing.assert_array_equal(obs.legacy().users, oo.users)
    np.testing.assert_array_equal(state, ss)
    actor = DiffusionActor(cfg)
    action, _ = actor.sample(obs)
    e.step(action)
    old.step(action.movement)
    for field in ("uav_xy", "user_xy", "Y", "uplink", "history"):
        np.testing.assert_array_equal(getattr(e, field), getattr(old, field))
    snapshot = e.snapshot()
    clone = ServiceEnvironment(cfg.env)
    clone.restore(snapshot)
    nested_equal(clone.snapshot(), snapshot)
    c, history, zero = e.C.copy(), e.history.copy(), e.zero_coverage_steps
    e._refresh(); e._refresh()
    np.testing.assert_array_equal(e.C, c)
    np.testing.assert_array_equal(e.history, history)
    assert e.zero_coverage_steps == zero
    next_action, _ = actor.sample(e.obs)
    first, second = e.step(next_action), clone.step(next_action)
    np.testing.assert_array_equal(first[0].users, second[0].users)
    assert first[2] == second[2]
    assert 0 <= e.info['robustness'] <= (1-1/cfg.env.num_uavs)*e.info['coverage']+1e-12


def test_bad_action_mapping_rejected(cfg):
    e = ServiceEnvironment(cfg.env)
    o, _, _ = e.reset(0)
    a, _ = DiffusionActor(cfg).sample(o)
    bad = copy.deepcopy(a)
    bad.offsets[-1] += 1
    with pytest.raises(ValueError, match="match"):
        e.step(bad)


def test_numpy_numba_and_terminal_history(cfg):
    other = copy.deepcopy(cfg.env)
    other.backend = "numpy"
    a, b = ServiceEnvironment(cfg.env), ServiceEnvironment(other)
    o, _, _ = a.reset(41)
    b.reset(41)
    actor = DiffusionActor(cfg)
    for _ in range(cfg.env.horizon):
        action, _ = actor.sample(o)
        x, y = a.step(action), b.step(action)
        np.testing.assert_allclose(x[0].users, y[0].users, atol=1e-6)
        np.testing.assert_allclose(x[1], y[1], atol=1e-6)
        assert x[2] == pytest.approx(y[2], abs=1e-10)
        o = x[0]
    assert x[3] and y[3] and a.t == cfg.env.horizon
    with pytest.raises(RuntimeError, match="ended"):
        a.step(action)
    o, _, _ = a.reset(42)
    assert not a.proposals.any() and not a.last_scores.any()
    np.testing.assert_array_equal(o.fixed[:, -4:-2], 0)
    np.testing.assert_array_equal(o.fixed[:, -2], (a.B.min(1)/(cfg.env.num_uavs+1)).astype(np.float32))


@pytest.mark.parametrize("backend", ["batch", "process"])
def test_vector_scalar_backend_consistency(cfg, backend):
    vector = ServiceVector(cfg.env, 2, backend)
    envs = [ServiceEnvironment(cfg.env), ServiceEnvironment(cfg.env)]
    try:
        obs, _, _ = vector.reset([31, 32])
        for e, seed in zip(envs, [31, 32]):
            e.reset(seed)
        actor = DiffusionActor(cfg)
        action, _ = actor.sample(obs)
        result = vector.step(action)
        reference = [e.step(a) for e, a in zip(envs, action.split(cfg.env.num_uavs))]
        np.testing.assert_allclose(result[2], [r[2] for r in reference], rtol=1e-6)
        np.testing.assert_array_equal(result[0].users, stack([r[0] for r in reference]).users)
        snap = vector.snapshot()
        vector.restore(snap)
        nested_equal(vector.snapshot(), snap)
    finally:
        vector.close()


def test_probability_recompute_hand_sum_and_gradients(cfg):
    seed_all(1)
    obs = synthetic(cfg)
    p = DiffusionActor(cfg)
    action, chain = p.sample(obs)
    assert np.all(np.linalg.norm(action.movement, axis=-1) < 1+1e-6)
    assert np.all((action.scores >= 0) & (action.scores <= 1))
    recomputed = p.evaluate_chain(obs, chain)
    torch.testing.assert_close(recomputed, torch.tensor(chain.logp), atol=5e-4, rtol=1e-5)
    torch.testing.assert_close((recomputed-torch.tensor(chain.logp)).exp(), torch.ones_like(recomputed), atol=5e-4, rtol=1e-5)
    enc = p.encode(obs.tensor('cpu'))
    mov, scores = torch.tensor(chain.movement), torch.tensor(chain.scores)
    step = cfg.diffusion.steps-1
    mm, ms = p.transition(enc, mov[:, 0], scores[:, 0], step)
    std = p.transition_std[step]
    manual = torch.distributions.Normal(mm[1], std).log_prob(mov[1, 1]).sum()
    manual += torch.distributions.Normal(ms[:2], std).log_prob(scores[:2, 1]).sum()
    torch.testing.assert_close(recomputed[1, 0], manual)
    advantages = torch.tensor([1., -.4, .2, -.7])[:, None]
    (-recomputed*advantages).mean().backward()
    for module in (p.ue, p.movement_head, p.score_head):
        assert any(x.grad is not None and x.grad.norm()>0 for x in module.parameters())


def test_empty_sets_and_permutation_equivariance(cfg):
    p = DiffusionActor(cfg)
    empty = synthetic(cfg, (0, 0, 0, 0), (0, 0, 0, 0))
    a, chain = p.sample(empty)
    assert len(a.scores) == 0 and chain.logp.shape == (4, cfg.diffusion.steps)
    loss = p.evaluate_chain(empty, chain).mean()
    loss.backward()
    assert torch.isfinite(loss)
    obs = synthetic(cfg, (5,), (4,))
    order = np.array([3, 1, 0, 4, 2])
    permuted = ServiceObservation(obs.fixed.copy(), obs.users[order], obs.offsets.copy(),
                                 obs.ue_ids[order], obs.service_mask[order])
    old = p.encode(obs.tensor('cpu'))
    new = p.encode(permuted.tensor('cpu'))
    score = torch.tensor([.2, .3, .7, -.1])
    new_candidate_ids = permuted.candidate_ids
    score_perm = torch.tensor([int(np.flatnonzero(obs.candidate_ids == i)[0]) for i in new_candidate_ids])
    movement = torch.tensor([[.4, -.3]])
    m1, s1 = p.predict_noise(old, movement, score, 3)
    m2, s2 = p.predict_noise(new, movement, score[score_perm], 3)
    torch.testing.assert_close(m1, m2)
    torch.testing.assert_close(s1[score_perm], s2)


def collect_rollout(cfg, learner, count=3):
    env = ServiceEnvironment(cfg.env)
    obs, state, _ = env.reset(17)
    result = DppoRollout()
    for _ in range(count):
        action, chain = learner.actor.sample(obs)
        value = learner.values(state[None]).numpy()
        no, ns, r, done, trunc, _ = env.step(action)
        result.append(stack([obs]), chain, state=state[None], reward=np.array([r], np.float32),
            value=value, next_value=learner.values(ns[None]).numpy(),
            terminated=np.array([done]), truncated=np.array([trunc]))
        obs, state = no, ns
    return result


def test_value_space_popart_gae_and_optimizer_count(cfg):
    learner = DiffusionLearner(cfg, 'cpu')
    rollout = collect_rollout(cfg, learner)
    obs, chain, data = rollout.packed()
    t = torch.tensor(data['state'].reshape(-1, cfg.env.state_dim))
    original = learner.values(t).clone()
    learner.average_normalizer.update(torch.tensor([.1, .3, .9]), learner.value[-1], learner.value_opt)
    torch.testing.assert_close(original, learner.values(t), atol=1e-5, rtol=1e-5)
    r = torch.ones(3, 1)
    v = torch.zeros_like(r)
    nv = torch.full_like(r, 50)
    term = torch.tensor([[False], [False], [True]])
    ad, ret = gae(r, v, nv, term, .995, .95)
    assert ret[-1].item() == 1  # true terminal never bootstraps
    assert ret[0].item() > 1
    x = torch.tensor([.5, -.7], requires_grad=True)
    old, target = torch.tensor([.1, -.2]), torch.tensor([.9, .3])
    manual = torch.maximum(torch.nn.functional.huber_loss(x, target, reduction='none', delta=10.),
         torch.nn.functional.huber_loss(old+(x-old).clamp(-.2,.2), target, reduction='none', delta=10.)).mean()
    torch.testing.assert_close(mappo_value_loss(x, target, old, cfg.diffusion.value_clip), manual)
    metrics = learner.update(rollout)
    assert learner.optimizer_steps == cfg.algo.ppo_epochs*cfg.algo.ppo_minibatches
    assert np.isfinite(list(metrics.values())).all()
    actor_state = copy.deepcopy(learner.actor.state_dict())
    loss = learner.values(t).sum()  # values deliberately no_grad
    assert not loss.requires_grad
    learner.value_components(t)[0].sum().backward()
    nested_equal(actor_state, learner.actor.state_dict())


def test_microbatch_matches_full_gradient(cfg):
    seed_all(81)
    a = DiffusionLearner(cfg, 'cpu')
    rollout = collect_rollout(cfg, a, 4)
    state = copy.deepcopy(a.state_dict())
    ca, cb = copy.deepcopy(cfg), copy.deepcopy(cfg)
    ca.diffusion.microbatch_agents = 2
    cb.diffusion.microbatch_agents = 1000
    x, y = DiffusionLearner(ca, 'cpu'), DiffusionLearner(cb, 'cpu')
    x.load_state_dict(state); y.load_state_dict(state)
    seed_all(20); x.update(rollout)
    seed_all(20); y.update(rollout)
    nested_equal(x.actor.state_dict(), y.actor.state_dict(), atol=1e-4)
    nested_equal(x.value.state_dict(), y.value.state_dict(), atol=1e-5)


def make_teacher(cfg, path):
    d = cfg.as_dict()
    d.pop('diffusion')
    d['algo'].update(name='mappo', entropy_coef=.01, mappo_state_dependent_std=True)
    tc = config_from_dict(d)
    learner = make_learner(tc, 'cpu')
    atomic_save(path, dict(kind='model', observation_version='beacon-set-v2', config=tc.as_dict(),
                          learner=learner.state_dict(), env_steps=cfg.diffusion.teacher_step))


def test_complete_two_stage_resume_export_and_report(cfg, tmp_path):
    from uav_research.diffusion_dppo.data import collect_demonstrations, DemoDataset, pretrain
    from uav_research.diffusion_dppo.runner import train_online, export_actor, DeployedServiceActor
    from uav_research.diffusion_dppo.report import report
    cfg.diffusion.teacher_checkpoint = str(tmp_path/'teacher.pt')
    make_teacher(cfg, tmp_path/'teacher.pt')
    collect_demonstrations(cfg, tmp_path/'demo')
    ds = DemoDataset(tmp_path/'demo', 'fit', cfg)
    o, m, s = ds.batch([3, 0, 3, 33])
    np.testing.assert_array_equal(o.fixed[0], o.fixed[2])
    np.testing.assert_array_equal(m[0], m[2])
    pretrain(cfg, tmp_path/'demo', tmp_path/'pretrain_continuous')
    pretrain(cfg, tmp_path/'demo', tmp_path/'pretrain_resumed', max_batches=1)
    pretrain(cfg, tmp_path/'demo', tmp_path/'pretrain_resumed', resume=tmp_path/'pretrain_resumed/last.pt')
    before = load(tmp_path/'pretrain_continuous/last.pt')
    after = load(tmp_path/'pretrain_resumed/last.pt')
    nested_equal(before['actor'], after['actor'])
    nested_equal(before['optimizer'], after['optimizer'])
    assert before['batches'] == after['batches']
    pretrained = tmp_path/'pretrain_continuous/best.pt'
    train_online(cfg, tmp_path/'continuous', pretrained=pretrained)
    first = train_online(cfg, tmp_path/'resumed', pretrained=pretrained, stop_after_updates=1)
    train_online(cfg, tmp_path/'resumed', resume=first['full_checkpoint'])
    a = load(tmp_path/'continuous/last.pt')
    b = load(tmp_path/'resumed/last.pt')
    nested_equal(a['learner'], b['learner'])
    export_actor(tmp_path/'continuous/last.pt', tmp_path/'actor.pt')
    deployed = DeployedServiceActor(tmp_path/'actor.pt')
    learner = DiffusionLearner(cfg, 'cpu')
    learner.load_state_dict(a['learner'])
    env = ServiceEnvironment(cfg.env)
    obs, _, _ = env.reset(10000)
    seed_all(5); x = deployed.act(obs)
    seed_all(5); y = learner.act(obs)
    nested_equal(x.movement, y.movement)
    nested_equal(x.scores, y.scores)
    report(tmp_path/'continuous', tmp_path/'report', moment_repeats=2, include_baseline=False)
    old = (tmp_path/'report/comparison.json').read_bytes()
    report(tmp_path/'run_no_longer_present', tmp_path/'report', rebuild_only=True, include_baseline=False)
    assert old == (tmp_path/'report/comparison.json').read_bytes()
    assert (tmp_path/'report/figures/task_curves.svg').exists()


def test_artifact_isolation_and_atomic_failure(cfg, tmp_path):
    with pytest.raises(RuntimeError):
        with atomic_directory(tmp_path/'broken') as p:
            (p/'partial').write_text('incomplete')
            raise RuntimeError('disk failure')
    assert not (tmp_path/'broken').exists()
    atomic_save(tmp_path/'pretrain.pt', artifact(cfg, 'pretrain', actor={}))
    with pytest.raises(ValueError, match='phase'):
        load(tmp_path/'pretrain.pt', cfg, {'dppo_full'})
    changed = copy.deepcopy(cfg)
    changed.env.capacity = 11
    with pytest.raises(ValueError, match='experiment'):
        load(tmp_path/'pretrain.pt', changed)
    from uav_research.runtime import load_checkpoint
    with pytest.raises(ValueError, match='dppo commands'):
        load_checkpoint(tmp_path/'pretrain.pt')
