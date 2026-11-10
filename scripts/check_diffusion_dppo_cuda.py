"""Bounded CUDA correctness/resume check. Writes only to the explicit output."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch

from uav_research.config import load_config, save_config
from uav_research.runtime import rng_state, restore_rng, seed_all, atomic_save, atomic_json, memory_stats
from uav_research.diffusion_dppo.storage import configure, artifact, load
from uav_research.diffusion_dppo.environment import ServiceVector
from uav_research.diffusion_dppo.learner import DiffusionLearner, DppoRollout


def difference(left, right):
    if isinstance(left, torch.Tensor):
        return float((left.detach().cpu()-right.detach().cpu()).abs().max()) if left.numel() else 0.
    if isinstance(left, np.ndarray):
        return float(np.max(np.abs(left-right))) if left.size else 0.
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((difference(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        return max((difference(a, b) for a, b in zip(left, right)), default=0.)
    assert left == right, (left, right)
    return 0.


def rollout(learner, vector, obs, state, steps=2):
    data = DppoRollout()
    for _ in range(steps):
        action, chain = learner.actor.sample(obs)
        value = learner.values(state).cpu().numpy()
        no, ns, reward, done, truncated, _ = vector.step(action)
        data.append(obs, chain, state=state, reward=reward, value=value,
            next_value=learner.values(ns).cpu().numpy(), terminated=done, truncated=truncated)
        obs, state = no, ns
    return data, obs, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.run.num_envs, cfg.algo.ppo_epochs, cfg.algo.ppo_minibatches = 2, 1, 1
    cfg.run.total_steps = 16
    cfg.validate()
    save_config(cfg, output/'config.yaml')
    device = configure(cfg)
    if device.type != 'cuda':
        raise ValueError('This check must execute on CUDA')
    seed_all(991001)
    learner = DiffusionLearner(cfg, device)
    vector = ServiceVector(cfg.env, 2, 'batch')
    restored_vector = None
    try:
        obs, state, _ = vector.reset([990101, 990102])
        action, chain = learner.actor.sample(obs)
        recomputed = learner.actor.evaluate_chain(obs, chain)
        delta = float((recomputed-torch.as_tensor(chain.logp, device=device)).abs().max())
        ratio_error = float(((recomputed-torch.as_tensor(chain.logp, device=device)).exp()-1).abs().max())
        if delta > .005 or ratio_error > .005:
            raise AssertionError(f'CUDA probability recomputation mismatch: {delta}, ratio {ratio_error}')
        learner.actor_opt.zero_grad(set_to_none=True)
        (-recomputed.mean()).backward()
        gradients = {name: float(torch.nn.utils.get_total_norm([
            p.grad for p in module.parameters() if p.grad is not None]))
            for name, module in [('ue', learner.actor.ue), ('movement', learner.actor.movement_head),
                                 ('score', learner.actor.score_head)]}
        if not all(np.isfinite(x) and x > 0 for x in gradients.values()):
            raise AssertionError(gradients)
        learner.actor_opt.zero_grad(set_to_none=True)
        initial, obs, state = rollout(learner, vector, obs, state)
        metrics = learner.update(initial)
        saved = artifact(cfg, 'cuda_check', learner=learner.state_dict(), environments=vector.snapshot(),
            obs=obs.copy(), state=state.copy(), rng=rng_state())
        atomic_save(output/'resume.pt', saved)
        continuation, _, _ = rollout(learner, vector, obs, state)
        learner.update(continuation)
        expected = copy.deepcopy(learner.state_dict())
        saved = load(output/'resume.pt', cfg, {'dppo_cuda_check'})
        restored = DiffusionLearner(cfg, device)
        restored.load_state_dict(saved['learner'])
        restored_vector = ServiceVector(cfg.env, 2, 'batch')
        restored_vector.restore(saved['environments'])
        restore_rng(saved['rng'])
        repeated, _, _ = rollout(restored, restored_vector, saved['obs'], saved['state'])
        restored.update(repeated)
        resume_delta = difference(expected, restored.state_dict())
        if resume_delta > 1e-6:
            raise AssertionError(f'CUDA resumed training mismatch: {resume_delta}')
        terminal = torch.as_tensor(state[:1], device=device).repeat(2, 1)
        terminal[0, -1], terminal[1, -1] = 399/400, 1.
        factor = learner.horizon_factor(terminal)
        assert factor.tolist() == [1., 0.]
        assert learner.values(terminal)[1].item() == 0.
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                _, profile_chain = learner.actor.sample(obs)
                learner.actor.evaluate_chain(obs, profile_chain)
            prof.export_chrome_trace(str(output/'profile.json'))
            entries = [dict(key=e.key, calls=e.count, cpu_us=e.self_cpu_time_total,
                cuda_us=getattr(e, 'self_device_time_total', 0.)) for e in prof.key_averages()]
            atomic_json(output/'operators.json', sorted(entries, key=lambda e: e['cuda_us'], reverse=True))
        result = dict(passed=True, device=torch.cuda.get_device_name(device), torch=torch.__version__,
            cuda=torch.version.cuda, probability_max_abs_delta=delta, ratio_max_abs_error=ratio_error,
            gradient_norms=gradients, resume_max_abs_delta=resume_delta, update_metrics=metrics,
            joint_transitions=12, **memory_stats(device))
        atomic_json(output/'result.json', result)
        print(result, flush=True)
    finally:
        vector.close()
        if restored_vector:
            restored_vector.close()


if __name__ == '__main__':
    main()
