import torch
from torch.nn import functional as F

from uav_research.algorithms import mappo_value_loss
from uav_research.config import load_config


def test_default_matches_original_value_loss_and_gradient():
    pred = torch.tensor([0.6, -0.7, 15.0, 0.1], requires_grad=True)
    old = torch.zeros(4)
    target = torch.tensor([1.0, -2.0, -5.0, 0.0])
    original = torch.maximum(
        F.huber_loss(pred, target, reduction="none", delta=10.0),
        F.huber_loss(old + (pred - old).clamp(-0.2, 0.2),
                     target, reduction="none", delta=10.0),
    ).mean()
    actual = mappo_value_loss(pred, target, old, 0.2)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred)[0],
        torch.autograd.grad(original, pred)[0], rtol=0, atol=0,
    )


def test_unclipped_loss_retains_gradient_when_clipped_loss_is_flat():
    pred = torch.tensor([0.6], requires_grad=True)
    target, old = torch.ones(1), torch.zeros(1)
    clipped = mappo_value_loss(pred, target, old, 0.2, True)
    unclipped = mappo_value_loss(pred, target, old, 0.2, False)
    torch.testing.assert_close(unclipped, F.huber_loss(pred, target, delta=10.0))
    assert torch.autograd.grad(clipped, pred)[0].item() == 0
    torch.testing.assert_close(
        torch.autograd.grad(unclipped, pred)[0], torch.tensor([-0.4])
    )


def test_only_one_experimental_config_difference():
    base = load_config("configs/discrete_mappo_ablation_c_mobile_coverage.yaml")
    variant = load_config("configs/discrete_mappo_critic_no_value_clip.yaml")
    assert base.algo.value_clipping is True
    assert variant.algo.value_clipping is False
    variant.algo.value_clipping = True
    assert base.as_dict() == variant.as_dict()
