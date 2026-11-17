"""Frozen evaluation supplements for missing full-episode tail statistics.

No model/optimizer updates, no reserved test scenes, no source artifact writes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from uav_research.evaluate import policy_from_checkpoint, _evaluation_episodes
from uav_research.runtime import configure_runtime, isolated_rng, atomic_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def evaluate(checkpoint, output, device, action_seed=880001):
    checkpoint,output=Path(checkpoint),Path(output)
    if (output/'provenance.json').exists():
        p=json.loads((output/'provenance.json').read_text())
        assert p['sha256']==sha(checkpoint) and p['device']==device and p['action_seed']==action_seed
        print('ALREADY_COMPLETE',output,flush=True);return
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    before=sha(checkpoint);cfg,policy=policy_from_checkpoint(checkpoint,device)
    cfg.run.device=device;cfg.run.cpu_threads=8 if device=="cpu" else 4;cfg.run.numba_threads=4
    cfg.run.evaluation_policy='sample';cfg.run.evaluation_action_seed=action_seed
    assert list(cfg.run.validation_seeds)==list(range(10000,10032))
    assert not set(cfg.run.validation_seeds)&set(cfg.run.test_seeds)
    configure_runtime(cfg);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy.actor.eval().requires_grad_(False)
    names=['coverage','throughput_bps','reward'];data=[];scenes=[];started=time.perf_counter()
    with isolated_rng(),torch.inference_mode():
        torch.manual_seed(action_seed)
        for _,seed,infos,_,_ in _evaluation_episodes(cfg,policy,cfg.run.validation_seeds,0):
            assert len(infos)==cfg.env.horizon
            data.append([[i[k] for k in names] for i in infos]);scenes.append(seed)
            print('EVALUATED',output.name,seed,'coverage',np.mean([x['coverage'] for x in infos]),flush=True)
    assert sha(checkpoint)==before
    output.mkdir(parents=True,exist_ok=True)
    metrics=np.asarray(data,dtype=np.float64)
    np.savez_compressed(output/'step_metrics.npz',metrics=metrics,names=names,scene_seeds=scenes)
    delta=None
    path=checkpoint.parent/'validation.csv'
    if path.exists():
        with path.open() as f:matching=[r for r in csv.DictReader(f) if int(r['env_steps'])==policy.provenance['checkpoint_env_steps']]
        if matching:delta={k:float(metrics[:,:,j].mean()-float(matching[-1][k])) for j,k in enumerate(names[:2])}
    atomic_json(output/'provenance.json',dict(checkpoint=str(checkpoint),sha256=before,
        env_steps=policy.provenance['checkpoint_env_steps'],training_config=policy.provenance['training_config'],
        device=device,batch_size=cfg.run.num_envs,action_seed=action_seed,scene_seeds=scenes,
        evaluation_policy='sample',tf32=False,elapsed_seconds=time.perf_counter()-started,
        original_validation_difference=delta,source_unchanged=True,reason='Missing full-episode metric arrays; frozen evaluation only'))
    print('COMPLETE',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=['mappo','sac','sac_cpu'],required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    root=Path('analysis/tail100_all_completed_results_20261110/supplemental_evaluations')
    if a.group=='mappo':
        runs=[f'mappo_capacity10_neighbors{n}_1m_seed0' for n in [6,7,8]]+['mappo_coverage_throughput_capacity10_10m_seed0']
        for run in runs:evaluate(Path('runs')/run/'last.pt',root/run/'last',a.device)
    else:
        run='sac_coverage_time_factor_relative_central_history_server_1m_seed0'
        source=Path('runs')/run
        if not source.exists():source=Path('.cache/tail100_sources')/run
        for kind in ['best','last']:
            for seed in ([880001,880002,880003] if a.group=='sac_cpu' else [880001]):
                output=root/run/(kind+'_cpu_seed'+str(seed)) if a.group=='sac_cpu' else root/run/kind
                evaluate(source/(kind+'.pt'),output,a.device,seed)

if __name__=='__main__':main()
