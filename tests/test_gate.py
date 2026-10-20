import copy
import csv
import pytest
from uav_research.config import load_config
from uav_research.runtime import atomic_json
from uav_research.evaluate import check_gate


def fixtures(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    configs = [copy.deepcopy(cfg) for _ in range(3)]
    configs[1].algo.name = "mappo"
    configs[2].env.robustness_weight = 0.0
    values = [
        (0.8, 100.0, 0.65, 0.81),
        (0.7, 90.0, 0.55, 0.78),
        (0.8, 100.0, 0.6, 0.75),
    ]
    paths = []
    for i, (config, metrics) in enumerate(zip(configs, values)):
        path = tmp_path / str(i)
        path.mkdir()
        atomic_json(path / "evaluation_config.json", config.as_dict())
        atomic_json(
            path / "evaluation_source.json",
            dict(
                finite_training_metrics=True,
                training_config=config.as_dict(),
                training_manifest={"status": "completed", "env_steps": 256},
            ),
        )
        with (path / "episodes.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "scene_seed",
                    "coverage",
                    "throughput_bps",
                    "robustness",
                    "retention_ratio",
                ]
            )
            for seed in config.run.validation_seeds:
                writer.writerow([seed, *metrics])
        paths.append(path)
    return paths


def test_gate_requires_all_metrics_and_positive_ablation(tmp_path):
    paths = fixtures(tmp_path)
    report = check_gate(*paths, tmp_path / "gate.json")
    assert report["passed"] and report["training_steps"] == 256
    data = load_config("configs/smoke.yaml").as_dict()
    data["algo"]["name"] = "mappo"
    data["env"]["num_users"] = 25
    atomic_json(paths[1] / "evaluation_config.json", data)
    with pytest.raises(ValueError, match="environments differ"):
        check_gate(*paths, tmp_path / "gate.json")


def test_gate_rejects_unstable_training_and_missing_validation_scenes(tmp_path):
    paths = fixtures(tmp_path)
    atomic_json(
        paths[0] / "evaluation_source.json",
        dict(
            finite_training_metrics=False,
            training_config=load_config("configs/smoke.yaml").as_dict(),
            training_manifest={"status": "completed", "env_steps": 256},
        ),
    )
    assert not check_gate(*paths, tmp_path / "gate.json")["passed"]
    for path in paths:
        rows = path.joinpath("episodes.csv").read_text().splitlines()
        path.joinpath("episodes.csv").write_text("\n".join(rows[:2]) + "\n")
    with pytest.raises(ValueError, match="complete configured validation"):
        check_gate(*paths, tmp_path / "gate.json")
