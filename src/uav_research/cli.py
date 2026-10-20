from __future__ import annotations
import argparse
import json
from .config import load_config
from .runtime import configure_runtime, resolve_output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dual-gateway UAV research experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check-env", "benchmark", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/main.yaml")
        p.add_argument("--output", required=True)
        p.add_argument("--device", choices=["auto", "cpu", "cuda"])
        p.add_argument("--algo", choices=["sac", "mappo", "matd3"])
    p = sub.choices["check-env"]
    p.add_argument("--scenes", type=int, default=8)
    p.add_argument("--steps", type=int, default=32)
    p = sub.choices["benchmark"]
    p.add_argument("--counts", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["batch", "process"],
        default=["batch", "process"],
    )
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    p.add_argument("--steps", type=int, default=512)
    p = sub.choices["train"]
    p.add_argument("--steps", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume")
    p.add_argument("--lambda-rob", type=float)
    p.add_argument("--mu-disconnect", type=float, help="Weight of the disconnected-UAV fraction penalty")
    p.add_argument("--wall-seconds", type=float)
    p.add_argument("--hourly-price", type=float)
    p.add_argument("--budget", type=float)
    p = sub.choices["evaluate"]
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", choices=["static", "random", "heuristic"])
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--traces", type=int, default=2)
    p.add_argument(
        "--evaluation-lambda",
        type=float,
        help="Common reward weight when comparing lambda ablations",
    )
    p = sub.add_parser("calibrate")
    p.add_argument("--config", default="configs/main.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(30000, 30008)))
    p = sub.add_parser("export-actor")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("diagnose")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001])
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--lookahead", type=int, default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p = sub.add_parser("plot")
    p.add_argument("--output", required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--evaluations", nargs="*", default=[])
    p.add_argument("--traces", nargs="*", default=[])
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--calibration")
    p.add_argument(
        "--parameter",
        choices=[
            "robustness_weight",
            "num_users",
            "sinr_threshold_db",
            "uu_range",
            "ug_range",
            "num_bs",
        ],
    )
    p = sub.add_parser("gate")
    p.add_argument("--sac", required=True)
    p.add_argument("--mappo", required=True)
    p.add_argument("--ablation", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("suite")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=["pilot", "multiseed"], default="pilot")
    p.add_argument("--gate")
    p.add_argument("--include-matd3", action="store_true")
    args = parser.parse_args(argv)
    args.output = str(resolve_output(args.output))
    if args.command in ("train", "evaluate", "benchmark", "check-env"):
        cfg = load_config(args.config)
        if args.device:
            cfg.run.device = args.device
        if args.algo:
            cfg.algo.name = args.algo
    if args.command == "train":
        from .train import train

        for option, field in (
            ("steps", "total_steps"),
            ("num_envs", "num_envs"),
            ("seed", "seed"),
            ("wall_seconds", "max_wall_seconds"),
            ("hourly_price", "hourly_price"),
            ("budget", "budget"),
        ):
            if getattr(args, option) is not None:
                setattr(cfg.run, field, getattr(args, option))
        if args.lambda_rob is not None:
            cfg.env.robustness_weight = args.lambda_rob
        if args.mu_disconnect is not None:
            cfg.env.disconnection_weight = args.mu_disconnect
        result = train(cfg, args.output, args.resume)
    elif args.command == "evaluate":
        from .evaluate import evaluate, Baseline, policy_from_checkpoint

        device = configure_runtime(cfg)
        if bool(args.checkpoint) == bool(args.baseline):
            parser.error("Specify exactly one of --checkpoint or --baseline")
        if args.checkpoint:
            cfg, policy = policy_from_checkpoint(args.checkpoint, device)
            method = cfg.algo.name
        else:
            method = args.baseline
            policy = Baseline(cfg, method)
        if args.evaluation_lambda is not None:
            cfg.env.robustness_weight = args.evaluation_lambda
        seeds = args.seeds or (
            cfg.run.validation_seeds
            if args.split == "validation"
            else cfg.run.test_seeds
        )
        if args.split == "validation" and set(seeds) & set(cfg.run.test_seeds):
            parser.error("Test seeds cannot be used as validation")
        _, result = evaluate(cfg, policy, seeds, args.output, method, args.traces)
    elif args.command == "check-env":
        from .experiments import check_environment

        result = check_environment(cfg, args.output, args.scenes, args.steps)
    elif args.command == "benchmark":
        from .experiments import benchmark

        result = benchmark(
            cfg, args.output, args.counts, args.backends, args.devices, args.steps
        )
    elif args.command == "calibrate":
        from .calibration import calibrate

        result = calibrate(load_config(args.config), args.output, args.seeds)
    elif args.command == "export-actor":
        from .deployment import export_actor

        result = export_actor(args.checkpoint, args.output)
    elif args.command == "diagnose":
        from .experiments import diagnose

        if args.candidates < 2 or args.lookahead < 1:
            parser.error("candidates>=2 and lookahead>=1 required")
        result = diagnose(
            args.checkpoint,
            args.output,
            args.seeds,
            args.candidates,
            args.lookahead,
            args.device,
        )
    elif args.command == "plot":
        from .plotting import (
            curves,
            hierarchical_results,
            trace_plot,
            sensitivity,
            calibration_plots,
        )

        if args.calibration:
            calibration_plots(args.calibration, args.output)
        if not (args.runs or args.evaluations or args.traces or args.calibration):
            parser.error("Provide runs, evaluations or traces")
        if args.smooth < 1:
            parser.error("smooth must be >=1")
        if args.runs:
            curves(args.runs, args.output, args.smooth)
        if args.evaluations:
            if args.parameter:
                sensitivity(args.evaluations, args.parameter, args.output)
            else:
                hierarchical_results(args.evaluations, args.output)
        for trace in args.traces:
            trace_plot(trace, args.output)
        result = {"output": args.output}
    elif args.command == "suite":
        from .suite import run_suite

        result = run_suite(
            load_config(args.config),
            args.output,
            args.stage,
            args.gate,
            args.include_matd3,
        )
    else:
        from .evaluate import check_gate

        result = check_gate(args.sac, args.mappo, args.ablation, args.output)
    if args.command == "check-env":
        result = {
            "output": args.output,
            "passed": result["passed"],
            "scenes": len(result["scenes"]),
            "obs_shape": result["obs_shape"],
            "state_dim": result["state_dim"],
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
