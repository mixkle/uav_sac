"""Generate isolated DPPO configs from the current capacity-10 baseline.

--check is read-only and detects drift in any physical or learning parameter.
"""
import argparse
from pathlib import Path
from uav_research.config import load_config, save_config
from uav_research.diffusion_dppo.config import from_dict, baseline_differences


def configurations(baseline):
    data = baseline.as_dict()
    data["algo"].update(name="diffusion_dppo", actor_lr=1e-4, ppo_clip=.1,
                         entropy_coef=0., mappo_state_dependent_std=False)
    data["diffusion"] = {}
    main = from_dict(data)
    if baseline_differences(main, baseline):
        raise AssertionError("DPPO physical settings differ from baseline")
    server = from_dict(main.as_dict())
    server.run.device, server.run.cpu_threads, server.run.numba_threads = "cuda:0", 4, 8
    # RTX 3090 deployment benchmark: identical 2048-transition rollout,
    # 40 PPO updates took 19.59s at 512 agents and 4.66s at 8192 agents.
    # This changes gradient accumulation only, not the PPO minibatch size.
    server.diffusion.microbatch_agents = 8192
    server.diffusion.teacher_checkpoint = "/home/dym/uav_sac/runs/mappo_coverage_throughput_capacity10_1m_seed0/best.pt"
    server.diffusion.pretrained_checkpoint = "/home/dym/uav_sac/runs/diffusion_dppo_capacity10_coverage_throughput_1m_seed0/pretrain/best.pt"
    continued = from_dict(server.as_dict())
    continued.run.total_steps = 3_000_000
    continued.diffusion.microbatch_agents = 16384
    continued_5m = from_dict(continued.as_dict())
    continued_5m.run.total_steps = 5_000_000
    continued_10m = from_dict(continued_5m.as_dict())
    continued_10m.run.total_steps = 10_000_000
    dual = from_dict(continued.as_dict())
    dual.diffusion.actor_update_devices = ["cuda:0", "cuda:1"]
    # Total agents per accumulation wave across both devices, not PPO batch.
    dual.diffusion.microbatch_agents = 16384
    smoke = from_dict(main.as_dict())
    smoke.env.num_uavs, smoke.env.num_users, smoke.env.horizon = 4, 24, 8
    smoke.run.num_envs, smoke.run.cpu_threads, smoke.run.numba_threads = 2, 2, 2
    smoke.run.total_steps = smoke.run.eval_interval = 16
    smoke.run.validation_seeds, smoke.run.test_seeds = [10000, 10001], [20000]
    smoke.run.tensorboard = False
    smoke.algo.ppo_epochs = smoke.algo.ppo_minibatches = 2
    smoke.algo.rollout_steps = 4
    smoke.diffusion.pretrain_epochs, smoke.diffusion.pretrain_batch = 2, 32
    smoke.diffusion.fit_seeds, smoke.diffusion.selection_seeds = [300000, 300001], [301000]
    smoke.diffusion.minimum_free_gib = 1.
    probe = from_dict(main.as_dict())
    probe.run.total_steps = probe.run.eval_interval = 2048
    probe.run.validation_seeds, probe.run.test_seeds = [10000, 10001], [20000]
    probe.run.tensorboard = False
    probe.diffusion.fit_seeds, probe.diffusion.selection_seeds = [300000, 300001], [301000]
    probe.diffusion.pretrain_epochs = 1
    return {
        "diffusion_dppo_capacity10_coverage_throughput_1m.yaml": main,
        "diffusion_dppo_capacity10_coverage_throughput_1m_server.yaml": server,
        "diffusion_dppo_capacity10_coverage_throughput_3m_server.yaml": continued,
        "diffusion_dppo_capacity10_coverage_throughput_3m_server_dual.yaml": dual,
        "diffusion_dppo_capacity10_coverage_throughput_5m_server.yaml": continued_5m,
        "diffusion_dppo_capacity10_coverage_throughput_10m_server.yaml": continued_10m,
        "diffusion_dppo_local_smoke.yaml": smoke,
        "diffusion_dppo_local_probe.yaml": probe,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="configs/mappo_coverage_throughput_capacity10_1m.yaml")
    parser.add_argument("--output", default="configs")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, cfg in configurations(load_config(args.baseline)).items():
        cfg.validate()
        path = Path(args.output)/name
        if args.check:
            if load_config(path).as_dict() != cfg.as_dict():
                raise ValueError(f"Configuration drift detected: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_config(cfg, path)
        print(("checked " if args.check else "generated ")+str(path))


if __name__ == "__main__":
    main()
