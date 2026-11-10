"""Frozen-policy, same-transition SINR-score audit; shadow associations never feed back."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from uav_research.runtime import atomic_json, isolated_rng, seed_all, source_digest
from uav_research.topk_diagnostic import file_hash, atomic_npz
from uav_research.diffusion_dppo.storage import load, configure, Progress
from uav_research.diffusion_dppo.config import from_dict
from uav_research.diffusion_dppo.policy import DiffusionActor
from uav_research.diffusion_dppo.environment import ServiceVector, resolve_topk


@torch.no_grad()
def audit(checkpoint, output, device):
    checkpoint, output = Path(checkpoint), Path(output)
    data = load(checkpoint, kinds={'dppo_model'})
    digest = file_hash(checkpoint)
    cfg = from_dict(data['config'])
    cfg.run.device = device
    selected = configure(cfg)
    actor = DiffusionActor(cfg).to(selected)
    actor.load_state_dict(data['learner']['actor'])
    actor.eval()
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(30)
    keys = ['coverage', 'sinr_score_coverage', 'throughput_bps', 'sinr_score_throughput_bps',
            'reward', 'sinr_score_reward', 'changed_owner_fraction']
    metrics = []
    with isolated_rng():
        seed_all(cfg.run.evaluation_action_seed)
        for start in range(0, len(cfg.run.validation_seeds), cfg.run.num_envs):
            seeds = cfg.run.validation_seeds[start:start+cfg.run.num_envs]
            vector = ServiceVector(cfg.env, len(seeds), 'batch')
            try:
                obs, _, _ = vector.reset(seeds)
                rows = []
                for t in range(cfg.env.horizon):
                    before, scores = [], []
                    for env in vector.envs:
                        before.append(env.sensed & (env.Y >= cfg.env.sinr_threshold_db) &
                            (env.B.min(1) <= cfg.env.num_uavs)[:, None])
                        scores.append(1/(1+np.exp(-(env.Y-cfg.env.sinr_threshold_db)/10)))
                    action, _ = actor.sample(obs)
                    obs, _, _, _, _, infos = vector.step(action)
                    step = []
                    for i, env in enumerate(vector.envs):
                        saved_c, saved_history = env.C.copy(), env.history.copy()
                        after = (env.sensed & (env.Y >= cfg.env.sinr_threshold_db) &
                            (env.B.min(1) <= cfg.env.num_uavs)[:, None])
                        c, _, _ = resolve_topk(before[i], after, scores[i], cfg.env.capacity)
                        coverage = float(c.any(0).mean())
                        throughput = float((c*env.rates).sum())
                        reward = coverage + cfg.env.throughput_weight*throughput/(cfg.env.num_users*cfg.env.rate_reference_bps)
                        old_owner = np.where(env.C.any(0), env.C.argmax(0), -1)
                        new_owner = np.where(c.any(0), c.argmax(0), -1)
                        step.append([infos[i]['coverage'], coverage, infos[i]['throughput_bps'],
                            throughput, infos[i]['reward'], reward, float((old_owner != new_owner).mean())])
                        np.testing.assert_array_equal(env.C, saved_c)
                        np.testing.assert_array_equal(env.history, saved_history)
                    rows.append(step)
                    progress(f'score audit checkpoint={data["env_steps"]} scenes={seeds[0]}..{seeds[-1]} t={t+1}')
                metrics.append(np.asarray(rows).transpose(1, 0, 2))
            finally:
                vector.close()
    metrics = np.concatenate(metrics)
    atomic_npz(output/'step_metrics.npz', metrics=metrics, names=np.asarray(keys), seeds=cfg.run.validation_seeds)
    episodes = [dict(scene_seed=s, **dict(zip(keys, metrics[i].mean(0)))) for i, s in enumerate(cfg.run.validation_seeds)]
    with (output/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader(); writer.writerows(episodes)
    rng = np.random.default_rng(950001)
    draws = rng.integers(len(episodes), size=(2000, len(episodes)))
    paired = {}
    for original, shadow in [('coverage', 'sinr_score_coverage'), ('throughput_bps', 'sinr_score_throughput_bps'),
                             ('reward', 'sinr_score_reward')]:
        difference = metrics[:, :, keys.index(shadow)].mean(1)-metrics[:, :, keys.index(original)].mean(1)
        ci = np.quantile(difference[draws].mean(1), [.025, .975])
        paired[original] = dict(difference=float(difference.mean()), low=float(ci[0]), high=float(ci[1]))
    assert file_hash(checkpoint) == digest
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, env_steps=data['env_steps'],
        source_hash=source_digest(), scenes=cfg.run.validation_seeds, action_seed=cfg.run.evaluation_action_seed,
        device=device, metrics=dict(zip(keys, metrics.mean((0, 1)).tolist())), paired=paired,
        interpretation='Same physical transitions; SINR scores from pre-action information. '
        'Shadow association does not change future observations, movement or history. This is not closed-loop performance.')
    atomic_json(output/'summary.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    audit(args.checkpoint, args.output, args.device)
