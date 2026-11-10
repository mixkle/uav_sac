"""Scene-chunked demonstrations, memory-mapped sampling and resumable BC."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..config import config_from_dict, save_config
from ..algorithms import make_learner
from ..runtime import (seed_all, rng_state, restore_rng, isolated_rng,
                       atomic_json, atomic_save, load_checkpoint, source_digest, CSVLogger, memory_stats)
from ..topk_diagnostic import file_hash
from .config import contract
from .environment import ServiceEnvironment
from .observations import ServiceObservation, ServiceAction, stack, offsets
from .policy import DiffusionActor
from .storage import artifact, load, ensure_budget, save_arrays, Progress, configure as configure_runtime


def teacher_policy(cfg, device):
    path = Path(cfg.diffusion.teacher_checkpoint)
    data = load_checkpoint(path)
    teacher_cfg = config_from_dict(data["config"])
    if data["env_steps"] != cfg.diffusion.teacher_step:
        raise ValueError("Teacher actual checkpoint step differs from configured provenance")
    from dataclasses import asdict
    expected, actual = asdict(cfg.env), asdict(teacher_cfg.env)
    for key in ("motion_scaling", "history_motion_source"):
        expected.pop(key); actual.pop(key)
    if expected != actual or teacher_cfg.algo.name != "mappo" or teacher_cfg.algo.actor_sharing != "shared":
        raise ValueError("Teacher must be shared MAPPO with identical physical/observation config")
    learner = make_learner(teacher_cfg, device)
    learner.load_state_dict(data["learner"])
    learner.actor.eval()
    learner.actor.requires_grad_(False)
    return learner.actor, dict(path=str(path), step=data["env_steps"], sha256=file_hash(path))


def teacher_action(actor, obs, cfg, previous_command=None):
    with torch.no_grad():
        device = next(actor.parameters()).device
        legacy = obs.legacy()
        if cfg.env.history_motion_source != "command":
            if previous_command is None:
                raise ValueError("Executed-history teacher adapter requires previous raw command")
            legacy.fixed[..., -4:-2] = previous_command
        movement, _, latent = actor.sample(legacy.to_torch(device), False)
    # Radio feature already normalized; undo it before constructing score logits.
    target_s = (obs.users[obs.service_mask, 3]*cfg.env.sinr_scale_db-cfg.env.sinr_threshold_db)/cfg.diffusion.teacher_score_scale_db
    score = torch.as_tensor(target_s, dtype=torch.float64).sigmoid().numpy()
    action = ServiceAction(movement.cpu().numpy(), score, obs.candidate_ids.copy(), offsets(obs.candidate_lengths))
    return action, latent.cpu().numpy().reshape(-1, 2), target_s.astype(np.float32)


def collect_demonstrations(cfg, output):
    cfg.validate()
    device = configure_runtime(cfg)
    output = Path(output)
    # Worst-case all UE visible, observations + IDs + masks + score labels.
    episodes = len(cfg.diffusion.fit_seeds)+len(cfg.diffusion.selection_seeds)
    estimate = episodes*cfg.env.horizon*cfg.env.num_uavs*(cfg.env.fixed_obs_dim*4+
                cfg.env.num_users*(5*4+8+1+4)+32)
    ensure_budget(cfg, output, estimate)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.diffusion.teacher_action_seed)
    teacher, source = teacher_policy(cfg, device)
    source_before = source["sha256"]
    info = dict(contract=contract(cfg), teacher=source, source_hash=source_digest(),
                fit_seeds=cfg.diffusion.fit_seeds, selection_seeds=cfg.diffusion.selection_seeds,
                expected_joint_transitions=episodes*cfg.env.horizon, reset="baseline_greedy_then_topk",
                complete=False)
    if (output/"dataset.json").exists():
        old = json.loads((output/"dataset.json").read_text())
        if old["contract"] != info["contract"] or any(old["teacher"][k] != source[k] for k in ("sha256", "step")):
            raise ValueError("Existing demonstrations belong to another experiment/teacher")
        if old.get("complete"):
            # Validate every stored chunk but preserve the dataset manifest hash.
            DemoDataset(output, "fit", cfg)
            DemoDataset(output, "selection", cfg)
            return old
    atomic_json(output/"dataset.json", info)
    save_config(cfg, output/"config.yaml")
    progress = Progress(cfg.diffusion.heartbeat_seconds)
    start = time.perf_counter()
    for split, seeds in (("fit", cfg.diffusion.fit_seeds), ("selection", cfg.diffusion.selection_seeds)):
        for i, scene in enumerate(seeds):
            path = output/split/f"scene_{scene}"
            if path.exists():
                meta = json.loads((path/"metadata.json").read_text())
                if meta["teacher_sha256"] != source_before:
                    raise ValueError("Demonstration source changed")
                for name, digest in meta["files"].items():
                    if file_hash(path/name) != digest:
                        raise ValueError("Demonstration checksum mismatch")
                continue
            progress(f"demonstrations {split} scene={scene} ({i+1}/{len(seeds)})", True)
            ensure_budget(cfg, output)
            env = ServiceEnvironment(cfg.env)
            obs, _, _ = env.reset(scene)
            observations, targets_m, targets_s, rows = [], [], [], []
            previous_command = np.zeros((cfg.env.num_uavs,2), np.float32)
            # Per-scene stream makes interrupted collection order-independent.
            with isolated_rng():
                seed_all(int(np.random.SeedSequence([cfg.diffusion.teacher_action_seed, scene]).generate_state(1)[0]))
                for t in range(cfg.env.horizon):
                    action, m, s = teacher_action(teacher, obs, cfg, previous_command)
                    previous_command = action.movement.copy()
                    observations.append(obs.copy())
                    targets_m.append(m)
                    targets_s.append(s)
                    obs, _, _, done, _, row = env.step(action)
                    rows.append([row[k] for k in ("reward", "coverage", "throughput_bps", "proposal_acceptance")])
                    progress(f"demonstrations {split} scene={scene} t={t+1}/{cfg.env.horizon}")
            whole = stack(observations)
            arrays = dict(fixed=whole.fixed.reshape(-1, cfg.env.fixed_obs_dim), users=whole.users,
                offsets=whole.offsets, ue_ids=whole.ue_ids, service_mask=whole.service_mask,
                candidate_offsets=offsets(whole.candidate_lengths), movement=np.concatenate(targets_m),
                scores=np.concatenate(targets_s), trajectory_metrics=np.asarray(rows))
            save_arrays(path, arrays, dict(scene=scene, groups=whole.groups, steps=len(rows),
                teacher_sha256=source_before, total_return=float(np.asarray(rows)[:, 0].sum())))
    if file_hash(cfg.diffusion.teacher_checkpoint) != source_before:
        raise ValueError("Frozen teacher file changed during demonstration collection")
    info.update(complete=True, elapsed_seconds=time.perf_counter()-start)
    atomic_json(output/"dataset.json", info)
    return info


class DemoDataset:
    def __init__(self, root, split, cfg):
        self.root, self.split = Path(root), split
        meta = json.loads((self.root/"dataset.json").read_text())
        if meta["contract"] != contract(cfg) or not meta["complete"]:
            raise ValueError("Incomplete or mismatched demonstration dataset")
        seeds = meta["fit_seeds"] if split == "fit" else meta["selection_seeds"]
        self.paths = [self.root/split/f"scene_{s}" for s in seeds]
        self.starts = offsets([json.loads((p/"metadata.json").read_text())["groups"] for p in self.paths])
        self.cache = OrderedDict()
        for p in self.paths:
            m = json.loads((p/"metadata.json").read_text())
            for name, sha in m["files"].items():
                if file_hash(p/name) != sha:
                    raise ValueError("Demonstration checksum mismatch")

    def __len__(self):
        return int(self.starts[-1])

    def _scene(self, i):
        if i not in self.cache:
            p = self.paths[i]
            arrays = {name: np.load(p/(name+".npy"), mmap_mode="r", allow_pickle=False)
                      for name in ("fixed", "users", "offsets", "ue_ids", "service_mask", "candidate_offsets", "movement", "scores")}
            self.cache[i] = arrays
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def epoch_order(self, rng):
        # Shuffle scene blocks and every sample inside each block. Every UAV
        # sample appears once, while mmap access stays bounded and sequential.
        return np.concatenate([self.starts[i]+rng.permutation(self.starts[i+1]-self.starts[i])
                               for i in rng.permutation(len(self.paths))])

    def batch(self, indices):
        indices = np.asarray(indices, np.int64)
        scene = np.searchsorted(self.starts[1:], indices, side="right")
        observations, movement, scores, original_positions = [], [], [], []
        for i in np.unique(scene):
            positions = np.flatnonzero(scene == i)
            local = indices[positions]-self.starts[i]
            a = self._scene(int(i))
            source = ServiceObservation(a["fixed"], a["users"], a["offsets"], a["ue_ids"], a["service_mask"])
            observations.append(source.take(local))
            movement.append(a["movement"][local])
            sr = np.concatenate([np.arange(a["candidate_offsets"][j], a["candidate_offsets"][j+1]) for j in local])
            scores.append(a["scores"][sr])
            original_positions.extend(positions)
        whole = ServiceObservation(np.concatenate([o.fixed for o in observations]),
            np.concatenate([o.users for o in observations]), offsets(np.concatenate([np.diff(o.offsets) for o in observations])),
            np.concatenate([o.ue_ids for o in observations]), np.concatenate([o.service_mask for o in observations]))
        inverse = np.argsort(original_positions)
        co = offsets(whole.candidate_lengths)
        score_rows = np.concatenate([np.arange(co[i], co[i+1]) for i in inverse])
        return whole.take(inverse), np.concatenate(movement)[inverse], np.concatenate(scores)[score_rows]


def selection_loss(actor, dataset, cfg):
    device = next(actor.parameters()).device
    generator = torch.Generator(device=device).manual_seed(cfg.diffusion.selection_noise_seed)
    total = np.zeros(2)
    with torch.no_grad():
        for lo in range(0, len(dataset), cfg.diffusion.pretrain_batch):
            ids = np.arange(lo, min(len(dataset), lo+cfg.diffusion.pretrain_batch))
            obs, m, s = dataset.batch(ids)
            step = torch.as_tensor(ids % actor.steps, device=device)
            mov, score = actor.denoising_loss(obs, m, s, generator=generator, steps=step)
            total += np.asarray([float(mov.sum()), float(score.sum())])
    return total / len(dataset)


def pretrain(cfg, demonstrations, output, resume=None, max_batches=None):
    cfg.validate()
    device = configure_runtime(cfg)
    output = Path(output)
    ensure_budget(cfg, output)
    output.mkdir(parents=True, exist_ok=True)
    fit, selection = DemoDataset(demonstrations, "fit", cfg), DemoDataset(demonstrations, "selection", cfg)
    seed_all(cfg.diffusion.pretrain_seed)
    actor = DiffusionActor(cfg).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.diffusion.pretrain_lr, eps=1e-5)
    order_rng = np.random.default_rng([cfg.diffusion.pretrain_seed, 401])
    provenance = {"dataset_sha256": file_hash(Path(demonstrations)/"dataset.json"),
                  "dataset_path": str(demonstrations), "source_hash": source_digest()}
    epoch, cursor, order, best, batches = 0, 0, None, float("inf"), 0
    partial_sum, partial_count = np.zeros(2), 0
    elapsed_before = 0.
    if resume:
        state = load(resume, cfg, {"dppo_pretrain"})
        if state["provenance"]["dataset_sha256"] != provenance["dataset_sha256"]:
            raise ValueError("Demonstrations changed during pretraining resume")
        actor.load_state_dict(state["actor"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, cursor, order, best, batches = [state[k] for k in ("epoch", "cursor", "order", "best", "batches")]
        partial_sum, partial_count = state["partial_sum"], state["partial_count"]
        elapsed_before = state.get("elapsed_seconds", 0.)
        order_rng.bit_generator.state = state["order_rng"]
        restore_rng(state["rng"])
    elif any((output/name).exists() for name in ("last.pt", "best.pt", "step0.pt")):
        raise FileExistsError("Pretraining output exists: use --resume")
    save_config(cfg, output/"config.yaml")
    progress = Progress(cfg.diffusion.heartbeat_seconds)
    start, last_save = time.perf_counter(), time.monotonic()
    fields = ["epoch", "batches", "fit_movement_mse", "fit_score_mse", "selection_movement_mse", "selection_score_mse", "selection_loss", "elapsed_seconds"]
    logger = CSVLogger(output/"history.csv", fields)

    def checkpoint():
        return artifact(cfg, "pretrain", actor=actor.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, cursor=cursor, order=order, best=best, batches=batches,
            order_rng=copy.deepcopy(order_rng.bit_generator.state), rng=rng_state(),
            partial_sum=partial_sum, partial_count=partial_count, provenance=provenance,
            elapsed_seconds=elapsed_before+time.perf_counter()-start)

    try:
        if not resume:
            sm, ss = selection_loss(actor, selection, cfg)
            best = cfg.diffusion.pretrain_mov_weight*sm+cfg.diffusion.pretrain_score_weight*ss
            if not np.isfinite(best):
                raise FloatingPointError("Nonfinite epoch-0 selection loss")
            atomic_save(output/"best.pt", checkpoint())
            atomic_save(output/"step0.pt", checkpoint())
        while epoch < cfg.diffusion.pretrain_epochs:
            if order is None:
                order = fit.epoch_order(order_rng)
            while cursor < len(order):
                ids = order[cursor:cursor+cfg.diffusion.pretrain_batch]
                obs, m, s = fit.batch(ids)
                optimizer.zero_grad(set_to_none=True)
                sums = np.zeros(2)
                for lo in range(0, len(ids), cfg.diffusion.microbatch_agents):
                    hi = min(len(ids), lo+cfg.diffusion.microbatch_agents)
                    co = offsets(obs.candidate_lengths)
                    mov, score = actor.denoising_loss(obs.take(np.arange(lo, hi)), m[lo:hi], s[co[lo]:co[hi]])
                    loss = (cfg.diffusion.pretrain_mov_weight*mov.sum()+cfg.diffusion.pretrain_score_weight*score.sum())/len(ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite pretraining loss")
                    loss.backward()
                    sums += [float(mov.detach().sum()), float(score.detach().sum())]
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.algo.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                partial_sum += sums
                partial_count += len(ids)
                cursor += len(ids)
                batches += 1
                progress(f"pretrain epoch={epoch+1}/{cfg.diffusion.pretrain_epochs} samples={cursor}/{len(order)} batches={batches}")
                if time.monotonic()-last_save >= cfg.run.checkpoint_minutes*60:
                    ensure_budget(cfg, output)
                    atomic_save(output/"last.pt", checkpoint())
                    last_save = time.monotonic()
                if max_batches is not None and batches >= max_batches:
                    atomic_save(output/"last.pt", checkpoint())
                    return dict(status="interrupted", epoch=epoch, batches=batches)
            sm, ss = selection_loss(actor, selection, cfg)
            selected_loss = float(cfg.diffusion.pretrain_mov_weight*sm+cfg.diffusion.pretrain_score_weight*ss)
            epoch += 1
            cursor, order = 0, None
            logger.write(dict(epoch=epoch, batches=batches, fit_movement_mse=partial_sum[0]/partial_count,
                fit_score_mse=partial_sum[1]/partial_count, selection_movement_mse=sm,
                selection_score_mse=ss, selection_loss=selected_loss,
                elapsed_seconds=elapsed_before+time.perf_counter()-start))
            partial_sum, partial_count = np.zeros(2), 0
            if not np.isfinite(selected_loss):
                raise FloatingPointError("Nonfinite pretraining selection loss")
            if selected_loss < best:
                best = selected_loss
                atomic_save(output/"best.pt", checkpoint())
            atomic_save(output/"last.pt", checkpoint())
            progress(f"pretrain epoch={epoch} selection_loss={selected_loss:.6g} best={best:.6g}", True)
            ensure_budget(cfg, output)
        from .evaluation import evaluate
        with isolated_rng():
            chosen = load(output/"best.pt", cfg, {"dppo_pretrain"})
            actor.load_state_dict(chosen["actor"])
            check = evaluate(cfg, actor, cfg.diffusion.selection_seeds, output/"execution_check", traces=2)
        result = dict(status="complete", epoch=epoch, batches=batches, best_selection_loss=float(best),
                      execution_check=check["summary"], provenance=provenance,
                      elapsed_seconds=elapsed_before+time.perf_counter()-start, **memory_stats(device))
        atomic_json(output/"summary.json", result)
        return result
    except BaseException as exc:
        atomic_json(output/"failure.json", dict(error=repr(exc), epoch=epoch, cursor=cursor, batches=batches))
        try:
            ensure_budget(cfg, output)
            atomic_save(output/"failure_context.pt", checkpoint())
        except (OSError, RuntimeError, MemoryError):
            pass  # Keep the original error and the previous published checkpoint.
        raise
    finally:
        logger.close()
