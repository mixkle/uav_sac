from dataclasses import replace
import copy
import math
import numpy as np
import pytest
import torch
from uav_research.actions import DISCRETE_DIRECTIONS, decode_actions, nearest_actions
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment
from uav_research.vector import VectorEnvironment
from uav_research.models import CategoricalActor
from uav_research.algorithms import gae
from uav_research.train import train
from uav_research.runtime import load_checkpoint
from uav_research.deployment import export_actor, DeployedActor
from uav_research.evaluate import Baseline, evaluate, policy_from_checkpoint
from uav_research.plotting import curves, trace_plot


def test_discrete_mapping_equal_speed_and_boundary_dynamics():
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    e = replace(cfg.env, num_uavs=9)
    env = UAVEnvironment(e)
    env.reset(41)
    env.uav_xy[:] = [200, 200]
    old_users = env.user_xy.copy()
    env.step(np.arange(9))
    expected = DISCRETE_DIRECTIONS * 10
    np.testing.assert_allclose(env.uav_xy - [200, 200], expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(expected[:8], axis=1), 10, atol=1e-6)
    assert not np.array_equal(old_users, env.user_xy)
    assert env.t == 1 and np.all(env.actual_displacement[8] == 0)
    env.uav_xy[:] = [0, 400]
    env.step(np.full(9, 4, np.int64))
    np.testing.assert_allclose(env.uav_xy, np.tile([10 / np.sqrt(2), 400 - 10 / np.sqrt(2)], (9, 1)), atol=1e-6)
    assert env.info["boundary_clip_fraction"] == 0
    assert env.info["boundary_reflection_fraction"] == 1
    assert env.trace_frame()["action_ids"].tolist() == [4] * 9
    saved = env.snapshot()
    env.step(np.full(9, 8, np.int64))
    env.restore(saved)
    np.testing.assert_array_equal(env.last_action_ids, saved["action_ids"])


@pytest.mark.parametrize(
    "actions",
    [np.zeros(4, float), np.ones(4, bool), [-1] * 4, [9] * 4, np.zeros((4, 2), int)],
)
def test_discrete_invalid_actions(actions):
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    env = UAVEnvironment(cfg.env)
    env.reset(0)
    with pytest.raises(ValueError):
        env.step(actions)


@pytest.mark.parametrize("backend", ["batch", "process"])
def test_vector_discrete_matches_scalar(backend):
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    e = replace(cfg.env, horizon=8)
    scalar = [UAVEnvironment(e) for _ in range(2)]
    for env, seed in zip(scalar, [41, 42]):
        env.reset(seed)
    with VectorEnvironment(e, 2, backend) as vector:
        vector.reset([41, 42])
        for t in range(8):
            actions = (np.arange(8).reshape(2, 4) + t) % 9
            result = vector.step(actions)
            for i, env in enumerate(scalar):
                expected = env.step(actions[i])
                np.testing.assert_array_equal(result[1][i], expected[1])
                assert result[2][i] == expected[2]
        assert result[3].all()


def test_categorical_probabilities_empty_sets_and_gradient():
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    env = UAVEnvironment(cfg.env)
    raw, _, _ = env.reset(43)
    actor = CategoricalActor(cfg.env.obs_dim, 32)
    for raw in [
        raw,
        type(raw)(raw.fixed, np.empty((0, 4), np.float32), np.zeros(5, np.int64)),
    ]:
        obs = raw.to_torch("cpu")
        dist = actor.distribution(obs)
        torch.testing.assert_close(dist.probs.sum(-1), torch.ones(4))
        actions, old, _ = actor.sample(obs)
        logp, entropy, maximum = actor.evaluate_actions(obs, actions)
        torch.testing.assert_close(logp, old)
        torch.testing.assert_close((logp - old).exp(), torch.ones(4))
        assert torch.all((entropy >= 0) & (entropy <= math.log(9) + 1e-6))
        assert torch.all(maximum >= 1 / 9 - 1e-6)
    for parameter in actor.net.parameters():
        parameter.data.zero_()
    assert actor.sample(obs, True)[0].tolist() == [0] * 4
    # Distribution at zero logits is uniform with exact categorical entropy.
    torch.testing.assert_close(
        actor.distribution(obs).entropy(), torch.full((4,), math.log(9))
    )
    actor = CategoricalActor(cfg.env.obs_dim, 32)
    raw, _, _ = env.set_positions(np.full((4, 2), 200.0), np.full((24, 2), 200.0))
    actor.evaluate_actions(raw.to_torch("cpu"), torch.zeros(4, dtype=torch.long))[
        0
    ].sum().backward()
    assert (
        sum(
            p.grad.abs().sum() for p in actor.encoder.parameters() if p.grad is not None
        )
        > 0
    )


def test_discrete_baselines_and_projection():
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    obs, _, _ = UAVEnvironment(cfg.env).reset(44)
    for name in ("static", "random", "heuristic"):
        actions = Baseline(cfg, name).act(obs)
        assert actions.shape == (4,) and actions.dtype.kind in "iu"
        decode_actions(actions, (4,))
        if name == "static":
            assert actions.tolist() == [8] * 4
    np.testing.assert_array_equal(nearest_actions(DISCRETE_DIRECTIONS), np.arange(9))
    assert nearest_actions([[0, 0.5]])[0] == 0  # equal distance to forward and hover
    cfg.algo.name = "sac"
    with pytest.raises(ValueError, match="requires MAPPO"):
        cfg.validate()


@pytest.mark.parametrize("cpu_threads", [1, 8])
def test_discrete_complete_pipeline_and_exact_resume(tmp_path, cpu_threads):
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    cfg.run.cpu_threads = cpu_threads
    cfg.run.total_steps = 64
    cfg.run.eval_interval = 32
    cfg.run.validation_seeds = [10000]
    full = train(cfg, tmp_path / "full")
    part = copy.deepcopy(cfg)
    part.run.total_steps = 32
    first = train(part, tmp_path / "split")
    resumed = train(cfg, tmp_path / "split", first["resume_checkpoint"])
    a = load_checkpoint(full["resume_checkpoint"])
    b = load_checkpoint(resumed["resume_checkpoint"])
    assert a["replay_metadata"] is None

    def exact(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                exact(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            for xx, yy in zip(x, y):
                exact(xx, yy)
        else:
            assert x == y

    exact(a["learner"], b["learner"])
    ecfg, policy = policy_from_checkpoint(
        tmp_path / "full" / "best.pt", torch.device("cpu")
    )
    rows, _ = evaluate(ecfg, policy, [10000], tmp_path / "evaluation", trace_count=1)
    assert np.isfinite(rows[0]["return"])
    export_actor(tmp_path / "full" / "best.pt", tmp_path / "actor.pt")
    deployed = DeployedActor(tmp_path / "actor.pt")
    obs, _, _ = UAVEnvironment(cfg.env).reset(10000)
    np.testing.assert_array_equal(deployed.act(obs), policy.act(obs, True))
    curves([tmp_path / "full"], tmp_path / "figures")
    trace_plot(tmp_path / "evaluation" / "trace_10000.npz", tmp_path / "figures")
    bad = copy.deepcopy(cfg)
    bad.env.action_space = "continuous"
    with pytest.raises(ValueError, match="action space"):
        train(bad, tmp_path / "wrong", full["resume_checkpoint"])


def test_discrete_gae_terminal_and_partial_rollout():
    adv, ret = gae(
        torch.tensor([[2.0], [3.0]]),
        torch.zeros(2, 1),
        torch.tensor([[5.0], [10.0]]),
        torch.tensor([[False], [True]]),
        0.9,
        0.95,
    )
    assert ret[-1].item() == 3
    assert adv[0].item() == pytest.approx(2 + 0.9 * 5 + 0.9 * 0.95 * 3)
    adv, _ = gae(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.full((1, 1), 10.0),
        torch.zeros(1, 1, dtype=torch.bool),
        0.9,
        0.95,
    )
    assert adv.item() == 10


def test_discrete_partial_budget_and_failed_update_preserves_safe_checkpoint(
    tmp_path, monkeypatch
):
    from uav_research.algorithms import MAPPO

    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    cfg.run.total_steps = 66
    cfg.run.eval_interval = 32
    cfg.run.validation_seeds = [10000]
    result = train(cfg, tmp_path / "partial")
    assert result["env_steps"] == 66 and result["updates"] == 5
    original = MAPPO.update

    def fail(self, rollout):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(MAPPO, "update", fail)
    cfg.run.total_steps = 80
    with pytest.raises(RuntimeError, match="injected"):
        train(cfg, tmp_path / "partial", result["resume_checkpoint"])
    assert not (tmp_path / "partial" / "resume_000000000080").exists()
    assert load_checkpoint(result["resume_checkpoint"])["env_steps"] == 66
    monkeypatch.setattr(MAPPO, "update", original)


def test_discrete_mapping_version_is_checked_on_deployment(tmp_path):
    from uav_research.runtime import atomic_save

    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    cfg.run.total_steps = 16
    cfg.run.validation_seeds = [10000]
    train(cfg, tmp_path / "run")
    export_actor(tmp_path / "run" / "best.pt", tmp_path / "actor.pt")
    data = load_checkpoint(tmp_path / "actor.pt")
    data["action_specification"]["action_mapping_version"] = "wrong-version"
    atomic_save(tmp_path / "actor.pt", data)
    with pytest.raises(ValueError, match="mapping"):
        DeployedActor(tmp_path / "actor.pt")


def test_reflection_overshoot_and_legacy_clip():
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    for mode, expected in [("reflect", [395, 200]), ("clip", [400, 200])]:
        env = UAVEnvironment(replace(cfg.env, uav_boundary=mode))
        env.reset(4)
        env.uav_xy[:] = [395, 200]
        env.step(np.full(cfg.env.num_uavs, 3, dtype=np.int64))
        np.testing.assert_allclose(env.uav_xy, np.tile(expected, (cfg.env.num_uavs, 1)))
    # Multiple crossings within one move still end inside the domain.
    env = UAVEnvironment(replace(cfg.env, max_speed=1210))
    env.reset(4)
    env.uav_xy[:] = [395, 200]
    env.step(np.full(cfg.env.num_uavs, 3, dtype=np.int64))
    np.testing.assert_allclose(env.uav_xy, np.tile([5, 200], (cfg.env.num_uavs, 1)))


def test_sampling_evaluation_repeatable_and_rng_isolated():
    from uav_research.algorithms import make_learner
    cfg = load_config("configs/discrete_mappo_smoke.yaml")
    cfg.env.horizon = 3
    learner = make_learner(cfg, "cpu")
    flags = []
    original = learner.act
    def act(obs, deterministic=False):
        flags.append(deterministic)
        return original(obs, deterministic)
    learner.act = act
    before = torch.get_rng_state().clone()
    first, _ = evaluate(cfg, learner, [10000, 10001])
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(12345)
    second, _ = evaluate(cfg, learner, [10000, 10001])
    assert flags and not any(flags)
    for a, b in zip(first, second):
        for key in a:
            if key != "inference_ms_per_uav":
                assert a[key] == b[key], key
