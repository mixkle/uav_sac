"""Time batch256 updates on local acceptance replay; never writes trained weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from uav_research.config import load_config
from uav_research.algorithms import make_learner
from uav_research.replay import JointReplay
from uav_research.checkpoints import restore_replay
from uav_research.runtime import configure_runtime, load_checkpoint, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checks',type=Path,default=Path('analysis/sac_full_reward_implementation_checks/main_short'))
    p.add_argument('--output',type=Path,default=Path('analysis/sac_full_reward_implementation_checks/batch256_timing.json'))
    args=p.parse_args();results={}
    for mode in ['direct','time_factor']:
        run=args.checks/f'{mode}_short'
        cfg=load_config(run/'config.yaml');cfg.algo.batch_size=256
        device=configure_runtime(cfg);learner=make_learner(cfg,device)
        manifest=json.loads((run/'manifest.json').read_text())
        path=manifest['resume_checkpoint'];checkpoint=load_checkpoint(path,'cpu',mmap=True)
        learner.load_state_dict(checkpoint['learner'])
        replay=JointReplay(cfg.algo.replay_capacity,(cfg.env.num_uavs,cfg.env.obs_dim),cfg.env.state_dim)
        restore_replay(path,replay,checkpoint['replay_metadata'])
        sampling,updates=[],[]
        for i in range(55):
            stamp=time.perf_counter();batch=replay.sample(256,device);s=time.perf_counter()-stamp
            stamp=time.perf_counter();metrics=learner.update(batch);u=time.perf_counter()-stamp
            if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite benchmark')
            if i>=5:sampling.append(s);updates.append(u)
        mean_s=float(np.mean(sampling));mean_u=float(np.mean(updates))
        phases=manifest['phase_seconds'];per_transition=(phases['environment']+phases['inference'])/2048
        estimate=60000*(mean_s+mean_u)+250000*per_transition
        results[mode]=dict(batch_size=256,cpu_threads=8,warmup_updates=5,timed_updates=50,
                           sample_mean_seconds=mean_s,update_mean_seconds=mean_u,
                           update_p10_p90=np.quantile(updates,[.1,.9]).tolist(),
                           estimated_formal_collection_and_update_minutes=estimate/60,
                           estimate_excludes_validation_and_checkpoint_io=True,
                           replay_source=str(run),benchmark_weights_discarded=True)
        print(mode,json.dumps(results[mode]),flush=True)
    atomic_json(args.output,results)


if __name__=='__main__':main()
