"""Change ONLY the total online budget, preserving the immutable 1M configs."""
import copy
import hashlib
import json
from pathlib import Path

import yaml
from uav_research.config import load_config
from uav_research.diffusion_dppo.config import contract

ROOT=Path(__file__).resolve().parents[1]


def build():
    target=ROOT/'configs/motion_only_sinr_topk_10m'
    target.mkdir(exist_ok=True)
    entries={}
    for group in ('M1','M2'):
        source=ROOT/f'configs/motion_only_sinr_topk_1m/{group}.yaml'
        original=load_config(source)
        data=copy.deepcopy(original.as_dict())
        data['run']['total_steps']=10_000_000
        path=target/(group+'.yaml')
        path.write_text(yaml.safe_dump(data,sort_keys=False))
        assert contract(original)==contract(load_config(path))
        entries[group]=dict(config=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=f'runs/motion_only_sinr_topk_10m/{group}_seed0',
            source_output=f'runs/motion_only_sinr_topk_1m/{group}_seed0',
            source_config=str(source.relative_to(ROOT)),
            source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_steps=1_000_000,default_physical_gpu=2 if group=='M1' else 3)
    manifest=dict(version='motion-only-sinr-topk-10m-continuation-v1',runs=entries,
        total_steps=10_000_000,additional_steps=9_000_000,
        resume='full model/optimizers/normalizers/environments/RNG/best selection; no reinitialization',
        source_runs_preserved=True)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':print(json.dumps(build(),indent=2))
