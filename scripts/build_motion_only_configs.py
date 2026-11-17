"""Independent M1/M2 runs; never overwrite the A--F suite or its results."""
import copy
import hashlib
import json
from pathlib import Path

import yaml
from uav_research.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def build():
    # D4 already implements Gaussian motion only + PRE-move SINR Top-K.
    base = load_config(ROOT / 'configs/system_ablation_1m/D4.yaml').as_dict()
    target = ROOT / 'configs/motion_only_sinr_topk_1m'
    target.mkdir(exist_ok=True)
    entries = {}
    for group, algorithm in [('M1', 'gaussian_service_mappo'), ('M2', 'diffusion_dppo')]:
        cfg = copy.deepcopy(base)
        cfg['algo']['name'] = algorithm
        cfg['diffusion']['ablation'].update(
            run_id=group, initialization='random', initial_checkpoint='',
            score_actions=False, association='sinr_topk',
            diffusion_init_mode='epsilon' if group == 'M1' else 'normal_reference_residual',
            bc_objective='nll' if group == 'M1' else 'denoising')
        cfg['diffusion'].update(teacher_checkpoint='', teacher_step=0,
                                pretrained_checkpoint='', actor_update_devices=[])
        path = target / (group + '.yaml')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        load_config(path)
        entries[group] = dict(config=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=f'runs/motion_only_sinr_topk_1m/{group}_seed0',
            default_physical_gpu=2 if group == 'M1' else 3)
    manifest = dict(version='motion-only-sinr-topk-1m-v1', runs=entries,
        source='configs/system_ablation_1m/D4.yaml', online_steps=1_000_000,
        initialization='random; no teacher, demonstration, BC or old optimizer',
        diffusion_initialization='normal-reference epsilon residual with zero output projection; no pretrained weights',
        association='PRE SINR -> local Top10 -> one motion -> POST gate -> UE selection; no refill',
        action_seeds=[880001, 880002, 880003])
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
