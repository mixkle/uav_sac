from pathlib import Path
import copy
import json
import numpy as np
import pytest
import torch
from uav_research.config import load_config
from uav_research.train import train
from uav_research.runtime import load_checkpoint
from uav_research.evaluate import evaluate, Baseline, policy_from_checkpoint
from uav_research.plotting import curves, hierarchical_results, trace_plot
from uav_research.suite import run_suite


@pytest.mark.parametrize("name", ["sac", "mappo", "matd3"])
def test_full_pipeline_and_resume_matches_uninterrupted(tmp_path, name):
    cfg = load_config("configs/smoke.yaml")
    cfg.algo.name = name
    cfg.algo.warmup_steps = 16
    cfg.run.total_steps = 64
    cfg.run.eval_interval = 32
    cfg.run.log_interval = 32
    cfg.run.validation_seeds = [10000]
    cfg.run.tensorboard = False
    full = tmp_path / "full"
    result = train(cfg, full)
    assert result["status"] == "completed"
    part = copy.deepcopy(cfg)
    part.run.total_steps = 32
    split = tmp_path / "split"
    first = train(part, split)
    result = train(cfg, split, first["resume_checkpoint"])
    a = load_checkpoint(result["resume_checkpoint"])
    b = load_checkpoint(full / "resume_000000000064")
    for key, value in a["learner"]["actor"].items():
        torch.testing.assert_close(value, b["learner"]["actor"][key], rtol=0, atol=0)

    def exact(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                exact(x, y)
        else:
            assert left == right

    exact(a["learner"], b["learner"])
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
    if a["replay_metadata"]:
        import json

        for item in json.loads(
            (Path(result["resume_checkpoint"]) / "index.json").read_text()
        ):
            x = torch.load(
                Path(result["resume_checkpoint"]) / item["name"], weights_only=False
            )
            y = torch.load(
                full / "resume_000000000064" / item["name"], weights_only=False
            )
            for key in x:
                torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
    ecfg, policy = policy_from_checkpoint(full / "best.pt", torch.device("cpu"))
    rows, summary = evaluate(
        ecfg, policy, [10000], tmp_path / "evaluation", trace_count=1
    )
    assert len(rows) == 1 and np.isfinite(summary["mean"]["return"])
    with pytest.raises(ValueError, match="full checkpoint"):
        train(cfg, tmp_path / "wrong", full / "best.pt")


def test_plot_artifacts_and_no_single_seed_ci(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg.run.total_steps = 32
    cfg.run.validation_seeds = [10000]
    cfg.run.tensorboard = False
    train(cfg, tmp_path / "run")
    evaluate(
        cfg,
        Baseline(cfg, "static"),
        [10000],
        tmp_path / "eval",
        method="static",
        trace_count=1,
    )
    curves([tmp_path / "run"], tmp_path / "figures")
    summaries = hierarchical_results(
        [tmp_path / "eval"], tmp_path / "figures", repeats=20
    )
    assert all(row["ci_low"] is None for row in summaries)
    trace_plot(tmp_path / "eval" / "trace_10000.npz", tmp_path / "figures")
    for name in (
        "sample_efficiency",
        "wall_efficiency",
        "method_comparison",
        "coverage_robustness_tradeoff",
        "topology_10000",
        "faults_10000",
    ):
        for ext in ("pdf", "svg", "png"):
            assert (tmp_path / "figures" / f"{name}.{ext}").stat().st_size > 1000


def test_multiseed_rejects_failed_gate(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="gate failed"):
        run_suite(cfg, tmp_path / "suite", "multiseed", path)


def test_baselines_use_local_input_and_reproduce_movement(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    for method in ("static", "random", "heuristic"):
        first, _ = evaluate(cfg, Baseline(cfg, method), [10000], method=method)
        second, _ = evaluate(cfg, Baseline(cfg, method), [10000], method=method)
        for key in first[0]:
            if key != "inference_ms_per_uav":
                assert first[0][key] == second[0][key]


def test_batched_evaluation_preserves_scenes_and_baseline_rng():
    cfg = load_config("configs/smoke.yaml")
    for method in ("static", "random", "heuristic"):
        cfg.run.num_envs = 1
        scalar, _ = evaluate(cfg, Baseline(cfg, method), [10000, 10001])
        cfg.run.num_envs = 2
        batch, _ = evaluate(cfg, Baseline(cfg, method), [10000, 10001])
        for a, b in zip(scalar, batch):
            for key in a:
                if key == "inference_ms_per_uav":
                    continue
                if isinstance(a[key], (int, float)):
                    assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-9)
                else:
                    assert a[key] == b[key]
