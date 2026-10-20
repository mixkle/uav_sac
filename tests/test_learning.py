import copy
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.models import (
    disk_transform,
    disk_inverse,
    GaussianActor,
    ValueNormalizer,
    mlp,
)
from uav_research.algorithms import SAC, MAPPO, MATD3, gae, soft_target
from uav_research.replay import JointReplay
from uav_research.observations import RawObservation, TensorObservation
from uav_research.checkpoints import save_full, restore_replay
from uav_research.runtime import load_checkpoint
from uav_research.runtime import seed_all, configure_runtime, rng_state, restore_rng


def synthetic_obs(shape):
    groups = int(np.prod(shape[:-1]))
    return RawObservation(
        np.random.randn(*shape).astype(np.float32),
        np.random.randn(groups * 2, 4).astype(np.float32),
        np.arange(groups + 1) * 2,
    )


def synthetic_batch(cfg, n=16):
    e = cfg.env
    return dict(
        obs=synthetic_obs((n, e.num_uavs, e.obs_dim)).to_torch("cpu"),
        next_obs=synthetic_obs((n, e.num_uavs, e.obs_dim)).to_torch("cpu"),
        state=torch.randn(n, e.state_dim),
        next_state=torch.randn(n, e.state_dim),
        actions=disk_transform(torch.randn(n, e.num_uavs, 2))[0],
        reward=torch.rand(n),
        terminated=torch.arange(n) % 3 == 0,
    )


def test_disk_jacobian_inverse_and_gradient():
    for values in [[0.0, 0.0], [0.4, -0.8], [8.0, -5.0]]:
        z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        action, log_det = disk_transform(z)
        jac = torch.autograd.functional.jacobian(lambda x: disk_transform(x)[0], z)
        torch.testing.assert_close(torch.logdet(jac), log_det)
        torch.testing.assert_close(disk_inverse(action), z)
        assert action.norm() < 1
    assert torch.autograd.gradcheck(
        lambda x: disk_transform(x)[0],
        (torch.randn(5, 2, dtype=torch.float64, requires_grad=True),),
    )


def test_transformed_density_and_per_agent_ppo_ratios():
    actor = GaussianActor(5, 16)
    obs = synthetic_obs((8, 3, 5)).to_torch("cpu")
    action, logp, latent = actor.sample(obs)
    reevaluated, _ = actor.evaluate_latent(obs, latent)
    torch.testing.assert_close(logp, reevaluated)
    ratio = (reevaluated - logp).exp()
    assert ratio.shape == (8, 3)
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    manual = actor.distribution(obs).log_prob(latent).sum(-1) + 2 * torch.log1p(
        latent.square().sum(-1)
    )
    torch.testing.assert_close(logp, manual)


def test_terminal_targets_and_gae_no_cross_episode_leak():
    target = soft_target(
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([-2.0, -2.0]),
        0.9,
        0.1,
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 11.18]))
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    done = torch.tensor([[False], [True], [False]])
    advantage, returns = gae(
        rewards,
        torch.zeros_like(rewards),
        torch.ones_like(rewards) * 10,
        done,
        1.0,
        1.0,
    )
    torch.testing.assert_close(advantage, torch.tensor([[13.0], [2.0], [110.0]]))
    # Truncation bootstraps but prevents advantage propagation across reset.
    truncated = torch.tensor([[True], [False], [False]])
    advantage, _ = gae(
        rewards,
        torch.zeros_like(rewards),
        torch.ones_like(rewards) * 10,
        done,
        1.0,
        1.0,
        truncated,
    )
    assert advantage[0].item() == 11


def test_popart_preserves_unnormalized_values():
    net = mlp(5, 1, 16)
    norm = ValueNormalizer()
    opt = torch.optim.Adam(net.parameters())
    x = torch.randn(20, 5)
    before = norm.denormalize(net(x)).detach()
    norm.update(torch.randn(100) * 20 + 100, net[-1], opt)
    after = norm.denormalize(net(x)).detach()
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["sac", "mappo", "matd3"])
def test_learner_update_and_exact_restore(name):
    cfg = load_config("configs/smoke.yaml")
    cfg.algo.name = name
    device = configure_runtime(cfg)
    seed_all(0)
    cls = {"sac": SAC, "mappo": MAPPO, "matd3": MATD3}[name]
    learner = cls(cfg, device)
    batch = synthetic_batch(cfg)
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    if name == "mappo":
        shape = (4, 2, cfg.env.num_uavs, cfg.env.obs_dim)
        obs = synthetic_obs(shape)
        state = np.random.randn(4, 2, cfg.env.state_dim).astype(np.float32)
        with torch.no_grad():
            action, logp, z = learner.actor.sample(obs.to_torch("cpu"))
        rollout = dict(
            obs=obs,
            state=state,
            latent=z.numpy(),
            logp=logp.numpy(),
            value=np.zeros((4, 2), np.float32),
            next_value=np.ones((4, 2), np.float32),
            reward=np.random.rand(4, 2).astype(np.float32),
            terminated=np.array([[0, 0], [0, 1], [0, 0], [1, 1]], bool),
            truncated=np.zeros((4, 2), bool),
        )
        batch = rollout
    else:
        batch = {
            k: v if isinstance(v, TensorObservation) else v.detach()
            for k, v in batch.items()
        }
    if name == "matd3":
        learner.update(batch)
        assert all(
            torch.equal(v, before[k]) for k, v in learner.actor.state_dict().items()
        )
    metrics = learner.update(batch)
    assert all(np.isfinite(list(metrics.values())))
    assert any(
        not torch.equal(v, before[k]) for k, v in learner.actor.state_dict().items()
    )
    if name == "sac":
        assert metrics["alpha"] < cfg.algo.initial_alpha
    saved = copy.deepcopy(learner.state_dict())
    random_state = rng_state()
    metrics1 = learner.update(batch)
    restored = cls(cfg, device)
    restored.load_state_dict(saved)
    restore_rng(random_state)
    metrics2 = restored.update(batch)
    for key in metrics1:
        assert metrics1[key] == metrics2[key]
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(
            value, restored.actor.state_dict()[key], rtol=0, atol=0
        )


def test_actor_update_receives_all_uav_branches():
    cfg = load_config("configs/smoke.yaml")
    cfg.algo.hidden = 8
    learner = SAC(cfg, torch.device("cpu"))
    obs = synthetic_obs((4, cfg.env.num_uavs, cfg.env.obs_dim)).to_torch("cpu")
    action, logp, _ = learner.actor.sample(obs)
    action.retain_grad()
    learner.critic.requires_grad_(False)
    q1, q2 = learner.critic(torch.randn(4, cfg.env.state_dim), action)
    (-torch.minimum(q1, q2).mean()).backward()
    assert (action.grad.abs().sum((0, 2)) > 0).all()


def test_replay_wraparound_and_rng_restore(tmp_path):
    replay = JointReplay(5, (2, 3), 4, 7)
    for start in [0, 3, 6]:
        value = np.arange(start, start + 3, dtype=np.float32)
        replay.add(
            RawObservation(
                np.broadcast_to(value[:, None, None], (3, 2, 3)).copy(),
                np.empty((0, 4), np.float32),
                np.zeros(7, np.int64),
            ),
            np.broadcast_to(value[:, None], (3, 4)),
            np.zeros((3, 2, 2), np.float32),
            value,
            RawObservation(
                np.zeros((3, 2, 3), np.float32),
                np.empty((0, 4), np.float32),
                np.zeros(7, np.int64),
            ),
            np.zeros((3, 4), np.float32),
            np.zeros(3, bool),
        )
    assert replay.size == 5 and replay.cursor == 4
    assert set(replay.arrays["reward"]) == set(range(4, 9))
    save_full(tmp_path / "full", {}, replay)
    saved = load_checkpoint(tmp_path / "full")
    expected = replay.sample(4, torch.device("cpu"))
    restored = JointReplay(5, (2, 3), 4)
    restore_replay(tmp_path / "full", restored, saved["replay_metadata"])
    actual = restored.sample(4, torch.device("cpu"))
    for k in expected:
        if isinstance(expected[k], TensorObservation):
            for key in ("fixed", "users", "lengths"):
                torch.testing.assert_close(
                    getattr(expected[k], key), getattr(actual[k], key)
                )
        else:
            torch.testing.assert_close(expected[k], actual[k])
