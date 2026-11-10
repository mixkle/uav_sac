"""Local acceptance only: short main-scenario SAC runs, never formal 250k.

Both configurations explicitly override warmup, batch, capacity and validation
scenes; formal configs on disk are untouched. Refuses to overwrite results.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.train import train
from uav_research.runtime import configure_runtime, seed_all, atomic_json, isolated_rng
from uav_research.algorithms import make_learner
from uav_research.evaluate import evaluate
from uav_research.environment import UAVEnvironment


def actor_digest(learner):
    h=hashlib.sha256()
    for k,v in learner.actor.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('analysis/sac_full_reward_implementation_checks/main_short'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    configs={m:load_config(f'configs/sac_full_reward_{m}_250k.yaml') for m in ['direct','time_factor']}
    initial={}
    for mode,cfg in configs.items():
        device=configure_runtime(cfg);seed_all(cfg.run.seed);model=make_learner(cfg,device)
        # Paired actor/physics at t=0, sampled evaluation on calibration scenes only.
        probe=copy.deepcopy(cfg);probe.run.validation_seeds=[30020,30021]
        with isolated_rng():
            rows,summary=evaluate(probe,model,probe.run.validation_seeds,args.output/f'initial_{mode}',trace_count=1)
        for section in ['mean','std_across_scenes']:
            summary[section].pop('inference_ms_per_uav',None)
        initial[mode]=dict(actor_sha256=actor_digest(model),summary=summary)
    assert initial['direct']==initial['time_factor']
    with np.load(args.output/'initial_direct'/'trace_30020.npz') as a, np.load(args.output/'initial_time_factor'/'trace_30020.npz') as b:
        for key in ['uav_xy','user_xy','B','reward']:
            np.testing.assert_array_equal(a[key],b[key])
    atomic_json(args.output/'initial_pairing.json',initial)
    records={}
    for mode,cfg in configs.items():
        cfg.run.total_steps=2048;cfg.run.validation_seeds=[30020]
        cfg.run.test_seeds=[30021];cfg.run.eval_interval=1024;cfg.run.log_interval=256
        cfg.run.tensorboard=False;cfg.algo.warmup_steps=256;cfg.algo.batch_size=64
        cfg.algo.replay_capacity=4096
        before=time.perf_counter();result=train(cfg,args.output/f'{mode}_short')
        manifest=json.loads((args.output/f'{mode}_short'/'manifest.json').read_text())
        assert manifest['status']=='completed' and manifest['updates']==448
        records[mode]=dict(manifest=manifest,wall_seconds=time.perf_counter()-before,
                           overrides=dict(total_steps=2048,warmup_steps=256,batch_size=64,
                                          replay_capacity=4096,validation_seeds=[30020],test_seeds=[30021],
                                          eval_interval=1024,log_interval=256,tensorboard=False),
                           formal_configs_untouched=True)
    atomic_json(args.output/'acceptance.json',dict(initial_pairing=initial,short_runs=records,formal_runs_started=False))
    print(json.dumps({k:dict(seconds=v['wall_seconds'],peak_rss_mib=v['manifest']['peak_rss_mib']) for k,v in records.items()},indent=2))


if __name__=='__main__':main()
