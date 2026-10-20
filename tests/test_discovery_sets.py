from dataclasses import replace
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.environment import UAVEnvironment, physical_links
from uav_research.observations import RawObservation, stack_observations
from uav_research.algorithms import SAC
from uav_research.runtime import configure_runtime


def test_launch_area_distribution_and_no_coverage_rejection():
    cfg = replace(load_config().env, uav_initialization="bs_launch")
    cfg.backend = "numpy"
    radii = []
    for seed in range(64):
        env = UAVEnvironment(cfg)
        env.reset(seed)
        xy = env.uav_xy.copy()
        xy[8:, 1] = cfg.height - xy[8:, 1]
        r2 = (xy**2).sum(1)
        assert (r2 <= cfg.launch_radius**2).all()
        assert env.direct[np.arange(16), np.repeat([0, 1], 8)].all()
        assert (env.B.min(1) == 1).all()
        radii.extend(r2 / cfg.launch_radius**2)
    assert np.mean(radii) == pytest.approx(0.5, abs=0.04)
    cfg.sensing_threshold_db = 1000
    env = UAVEnvironment(cfg)
    obs, _, info = env.reset(0)
    assert info["coverage"] == 0 and len(obs.users) == 0


def test_sensing_gate_and_capacity_do_not_truncate_observation():
    cfg = load_config("configs/smoke.yaml").env
    cfg = replace(
        cfg, num_users=30, capacity=1, sensing_threshold_db=-100, sinr_threshold_db=-100
    )
    env = UAVEnvironment(cfg)
    obs, _, _ = env.reset(4)
    assert (np.diff(obs.offsets) == 30).all()
    assert (env.C.sum(1) <= 1).all()
    _, _, _, _, loss, ul = physical_links(cfg, env.uav_xy, env.user_xy)
    np.testing.assert_allclose(ul, cfg.beacon_power_dbm - loss - cfg.beacon_noise_dbm)
    cfg.sensing_threshold_db = 1000
    env = UAVEnvironment(cfg)
    obs, _, info = env.reset(4)
    assert len(obs.users) == 0 and info["sensing_rejected_users"] == 30
    assert env.C.sum() == 0


def test_set_permutation_empty_groups_and_gradient_ownership():
    cfg = load_config("configs/smoke.yaml")
    device = configure_runtime(cfg)
    learner = SAC(cfg, device)
    env = UAVEnvironment(cfg.env)
    obs, state, _ = env.reset(9)
    fixed = obs.fixed[None].copy()
    u = cfg.env.num_uavs
    users = np.random.default_rng(1).normal(size=(6, 4)).astype(np.float32)
    obs = RawObservation(fixed, users, np.array([0, 3, 3, 5, 6]))
    permutation = np.r_[2, 0, 1, 4, 3, 5]
    other = RawObservation(fixed, users[permutation], obs.offsets)
    with torch.no_grad():
        a = learner.actor.sample(obs.to_torch(device), True)[0]
        b = learner.actor.sample(other.to_torch(device), True)[0]
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        enc = learner.actor.encoder(obs.to_torch(device))
        assert not enc[0, 1, -33:-1].any()
    empty = RawObservation(
        fixed, np.empty((0, 4), np.float32), np.zeros(u + 1, np.int64)
    )
    assert torch.isfinite(learner.actor.sample(empty.to_torch(device))[0]).all()
    actor_before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in learner.critic.state_dict().items()}
    action, logp, _ = learner.actor.sample(obs.to_torch(device))
    learner.critic.requires_grad_(False)
    q = learner.critic(torch.tensor(state[None]), action)[0]
    (-q.mean() + 0.01 * logp.sum(-1).mean()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in learner.actor.encoder.parameters()
    )
    learner.actor_opt.step()
    assert any(
        not torch.equal(v, actor_before[k])
        for k, v in learner.actor.state_dict().items()
    )
    assert all(
        torch.equal(v, critic_before[k]) for k, v in learner.critic.state_dict().items()
    )
    learner.critic.requires_grad_(True)
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    with torch.no_grad():
        next_action, next_logp, _ = learner.actor.sample(obs.to_torch(device))
        target = q.detach() - 0.01 * next_logp.sum(-1)
    learner.critic_opt.zero_grad()
    loss = (
        (learner.critic(torch.tensor(state[None]), action.detach())[0] - target)
        .square()
        .mean()
    )
    loss.backward()
    learner.critic_opt.step()
    assert all(torch.equal(v, before[k]) for k, v in learner.actor.state_dict().items())
    alpha_before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    learner.alpha_opt.zero_grad()
    (-learner.log_alpha * (next_logp.sum(-1).detach() - 8)).mean().backward()
    learner.alpha_opt.step()
    assert all(
        torch.equal(v, alpha_before[k]) for k, v in learner.actor.state_dict().items()
    )


def test_observation_stack_select_duplicate_and_reset():
    cfg = load_config("configs/smoke.yaml").env
    env = UAVEnvironment(cfg)
    a, _, _ = env.reset(1)
    b, _, _ = env.reset(2)
    both = stack_observations([a, b])
    selected = both[[1, 1, 0]]
    for i, x in enumerate([b, b, a]):
        np.testing.assert_array_equal(selected[i].users, x.users)
    both[0] = b
    np.testing.assert_array_equal(both[0].users, b.users)


def test_terminal_at_600_and_rejection_partition():
    from uav_research.config import Config

    cfg = Config()
    env = UAVEnvironment(cfg.env)
    env.reset(30123)
    for step in range(600):
        _, _, _, done, truncated, info = env.step(np.zeros((16, 2), np.float32))
        assert done == (step == 599)
        assert not truncated
        lost = sum(
            info[k]
            for k in (
                "sensing_rejected_users",
                "downlink_rejected_users",
                "backhaul_rejected_users",
                "capacity_rejected_users",
            )
        )
        assert lost + env.C.sum() == cfg.env.num_users
        assert not (env.C.astype(bool) & ~env.sensed).any()


def test_actor_export_does_not_require_global_user_count(tmp_path):
    from uav_research.config import load_config
    from uav_research.algorithms import make_learner
    from uav_research.runtime import atomic_save
    from uav_research.deployment import export_actor, DeployedActor
    from uav_research.observations import OBSERVATION_VERSION

    for name in ("sac", "mappo", "matd3"):
        cfg = load_config("configs/smoke.yaml")
        cfg.algo.name = name
        learner = make_learner(cfg, torch.device("cpu"))
        obs, _, _ = UAVEnvironment(cfg.env).reset(7)
        path = tmp_path / (name + ".pt")
        atomic_save(
            path,
            dict(
                kind="model",
                observation_version=OBSERVATION_VERSION,
                config=cfg.as_dict(),
                learner=learner.state_dict(),
                env_steps=0,
            ),
        )
        target = tmp_path / (name + "_actor.pt")
        export_actor(path, target)
        deployed = DeployedActor(target)
        assert "num_users" not in str(deployed.artifact.keys())
        assert "critic" not in deployed.artifact
        np.testing.assert_array_equal(learner.act(obs, True), deployed.act(obs, True))


def test_sensing_threshold_includes_equality():
    cfg = load_config("configs/smoke.yaml").env
    env = UAVEnvironment(cfg)
    env.reset(31)
    xy, ue = env.uav_xy.copy(), env.user_xy.copy()
    threshold = float(physical_links(cfg, xy, ue)[5][0, 0])
    cfg.sensing_threshold_db = threshold
    env.set_positions(xy, ue)
    assert env.sensed[0, 0]
    cfg.sensing_threshold_db = np.nextafter(threshold, np.inf)
    env.set_positions(xy, ue)
    assert not env.sensed[0, 0]


def test_critic_action_gradient_reaches_encoder_without_entropy():
    cfg = load_config("configs/smoke.yaml")
    configure_runtime(cfg)
    learner = SAC(cfg, torch.device("cpu"))
    obs, state, _ = UAVEnvironment(cfg.env).reset(1)
    learner.critic.requires_grad_(False)
    action = learner.actor.sample(obs.to_torch(torch.device("cpu")))[0]
    (-learner.critic(torch.tensor(state[None]), action[None])[0].mean()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in learner.actor.encoder.parameters()
    )
