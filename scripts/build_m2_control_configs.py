"""Four controlled M2 runs, preserving the original 1M -> 10M schedule."""
import copy
import hashlib
import json
from pathlib import Path

import yaml
from uav_research.config import load_config
from uav_research.diffusion_dppo.config import contract

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('M2-NoHop', 'M2-N5', 'M2-N7', 'M2-N9')


def build():
    source = ROOT/'configs/motion_only_sinr_topk_10m/M2.yaml'
    base = load_config(source).as_dict()
    target = ROOT/'configs/m2_motion_neighbor_controls_10m'
    target.mkdir(exist_ok=True)
    runs = {}
    for index, group in enumerate(GROUPS):
        data = copy.deepcopy(base)
        data['diffusion']['ablation']['run_id'] = group
        if group == 'M2-NoHop':
            data['env']['motion_scaling']['enabled'] = False
            data['diffusion']['ablation']['motion_rule'] = 'none'
        else:
            data['env']['neighbor_slots'] = int(group[4:])
        entry = dict(default_physical_gpu=index,
            default_cpu_affinity=f'{6*index}-{6*index+5}',
            output=f'runs/m2_motion_neighbor_controls_10m/{group}_seed0',
            stage1_output=f'runs/m2_motion_neighbor_controls_1m/{group}_seed0')
        for steps, suffix, prefix in ((10_000_000, '', ''), (1_000_000, '_stage1_1m', 'stage1_')):
            cfg_data = copy.deepcopy(data)
            cfg_data['run']['total_steps'] = steps
            path = target/f'{group}{suffix}.yaml'
            path.write_text(yaml.safe_dump(cfg_data, sort_keys=False))
            cfg = load_config(path)
            entry[prefix+'config'] = str(path.relative_to(ROOT))
            entry[prefix+'sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if steps == 10_000_000:
                entry['fixed_observation_dim'] = cfg.env.fixed_obs_dim
                entry['critic_state_dim'] = cfg.env.state_dim
                final = cfg
            else:
                assert contract(final) == contract(cfg)
        runs[group] = entry
    manifest = dict(version='m2-motion-neighbor-controls-10m-v1',
        reference_config=str(source.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_output='runs/motion_only_sinr_topk_10m/M2_seed0',
        reference_stage1_output='runs/motion_only_sinr_topk_1m/M2_seed0',
        online_steps=10_000_000, stage1_steps=1_000_000,
        initialization='random; normal_reference_residual; no BC or transferred weights',
        schedule='train fresh to 1M; full-state continuation to cumulative 10M automatically',
        action_seeds=[880001,880002,880003], runs=runs)
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
