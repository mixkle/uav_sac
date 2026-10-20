import csv
import json

import pytest

from uav_research.cli import main
from uav_research.config import load_config
from uav_research.runtime import load_checkpoint, resolve_output
from uav_research.train import train
from test_gate import fixtures
from uav_research.evaluate import check_gate


@pytest.mark.parametrize("change", ["reward", "hidden", "lr", "schedule"])
def test_gate_rejects_confounded_training_even_with_evaluation_override(
    tmp_path, change
):
    paths = fixtures(tmp_path)
    index = 1 if change == "reward" else 2
    source_path = paths[index] / "evaluation_source.json"
    source = json.loads(source_path.read_text())
    cfg = source["training_config"]
    if change == "reward":
        cfg["env"]["robustness_weight"] = 0.25
    elif change == "hidden":
        cfg["algo"]["hidden"] = 128
    elif change == "lr":
        cfg["algo"]["actor_lr"] = 0.01
    else:
        cfg["run"]["num_envs"] = 4
    source_path.write_text(json.dumps(source))
    if change != "reward":
        (paths[index] / "evaluation_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="reward weights|Single-factor"):
        check_gate(*paths, tmp_path / "gate.json")


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_and_suite_preserve_explicit_learning_rates(
    tmp_path, monkeypatch, explicit
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "algo:\n  actor_lr: 0.0001\n  critic_lr: 0.0002\n" if explicit else "{}"
    )
    expected = (1e-4, 2e-4) if explicit else (5e-4, 5e-4)
    seen = []

    def fake_train(cfg, *args):
        seen.append(
            (
                cfg.algo.name,
                cfg.algo.learning_rate("actor"),
                cfg.algo.learning_rate("critic"),
            )
        )
        return {"status": "stopped"}

    monkeypatch.setattr("uav_research.train.train", fake_train)
    main(
        [
            "train",
            "--config",
            str(config_path),
            "--algo",
            "mappo",
            "--output",
            str(tmp_path / "cli"),
        ]
    )
    assert seen[-1][1:] == expected
    cfg = load_config(config_path)
    import uav_research.suite as suite

    def suite_train(c, *args):
        seen.append(
            (c.algo.name, c.algo.learning_rate("actor"), c.algo.learning_rate("critic"))
        )
        return {"status": "stopped" if c.algo.name == "mappo" else "completed"}

    monkeypatch.setattr(suite, "train", suite_train)
    monkeypatch.setattr(suite, "policy_from_checkpoint", lambda *args: (cfg, None))
    monkeypatch.setattr(suite, "evaluate", lambda *args, **kwargs: None)
    suite.run_suite(cfg, tmp_path / "suite")
    assert seen[-1][0] == "mappo" and seen[-1][1:] == expected


def test_resume_best_is_self_contained_and_rollback_requires_new_directory(
    tmp_path, monkeypatch
):
    cfg = load_config("configs/smoke.yaml")
    cfg.run.total_steps = cfg.run.eval_interval = cfg.run.log_interval = 32
    cfg.run.tensorboard = False
    means = dict(
        coverage=0.8,
        discovery_fraction=0.9,
        disconnection_ratio=0.1,
        reward_primary=0.8,
        user_displacement_mean=1.0,
        uav_nearest_neighbor_mean=100.0,
        uav_close_neighbor_fraction=0.0,
        throughput_bps=1e8,
        robustness=0.6,
        retention_ratio=0.75,
        worst_fault_coverage=0.5,
    )
    means["return"] = 1.0
    monkeypatch.setattr(
        "uav_research.train.evaluate", lambda *args: ([], {"mean": means})
    )
    original = tmp_path / "original"
    first = train(cfg, original)
    checkpoint = load_checkpoint(first["resume_checkpoint"])
    assert checkpoint["best_model"]["env_steps"] == 0
    # Resume with only the full checkpoint: no sibling best.pt is available.
    import shutil

    portable = tmp_path / "portable"
    shutil.copytree(first["resume_checkpoint"], portable)
    cfg.run.total_steps = 64
    fresh = tmp_path / "fresh"
    train(cfg, fresh, portable)
    assert load_checkpoint(fresh / "best.pt")["env_steps"] == 0
    assert (
        json.loads((fresh / "best_selection.json").read_text())["validation_return"]
        == 1.0
    )
    train(cfg, original, portable)
    before = {
        name: (original / name).read_bytes()
        for name in ("manifest.json", "validation.csv", "best.pt", "config.yaml")
    }
    cfg.run.total_steps = 48
    with pytest.raises(ValueError, match="new output directory"):
        train(cfg, original, portable)
    for name, content in before.items():
        assert (original / name).read_bytes() == content
    branch = tmp_path / "branch"
    train(cfg, branch, portable)
    with (branch / "validation.csv").open() as file:
        assert [int(row["env_steps"]) for row in csv.DictReader(file)] == [48]


def test_data_disk_output_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_DATA_ROOT", str(tmp_path / "data"))
    assert resolve_output("runs/probe") == tmp_path / "data/runs/probe"
    assert resolve_output(tmp_path / "explicit") == tmp_path / "explicit"
    # Direct Python APIs use the same root as the command-line entry points.
    from uav_research.evaluate import Baseline, evaluate

    cfg = load_config("configs/smoke.yaml")
    evaluate(
        cfg, Baseline(cfg, "static"), [10000], "evaluations/probe", method="static"
    )
    assert (tmp_path / "data/evaluations/probe/episodes.csv").exists()
