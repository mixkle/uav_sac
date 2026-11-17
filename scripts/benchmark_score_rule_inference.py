#!/usr/bin/env python3
"""Frozen S1 microbenchmark, outside execution and score-selection RNG streams."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from uav_research.score_rule_experiment import source_actor,check_frozen,DEFAULT_OUTPUT
from uav_research.score_rule_diagnostic import quality
from uav_research.diffusion_dppo.observations import ServiceObservation
from uav_research.diffusion_dppo.environment import resolve_topk
from uav_research.runtime import resolve_output,atomic_json

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',default=DEFAULT_OUTPUT)
a=p.parse_args();root=resolve_output(a.output);rows=[]
for ck in ('best','last'):
    cfg,actor,meta=source_actor(ck,a.device)
    path=root/'s1'/ck/'880001/score_samples/b0_t050.npz'
    with np.load(path) as z:data={k:z[k] for k in z.files}
    obs=ServiceObservation(data['fixed'],data['users'],data['observation_offsets'],data['ue_ids'],data['service_mask'])
    generator=torch.Generator(device=a.device).manual_seed(993001)
    with torch.no_grad():
        actor.sample(obs,generator=generator)
        for before,after,y in zip(data['before'],data['after'],data['pre_sinr']):resolve_topk(before,after,quality(y),10)
        for number in (1,16):
            times=[]
            for _ in range(5):
                if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
                t=time.perf_counter()
                for j in range(number):actor.sample(obs,generator=generator)
                if str(a.device).startswith('cuda'):torch.cuda.synchronize(a.device)
                times.append(time.perf_counter()-t)
            rows.append(dict(checkpoint=ck,method=f'diffusion_{number}_joint_samples',
                median_seconds=float(np.median(times)),minimum_seconds=min(times),env_batch=len(data['before']),
                uav_groups=obs.groups,candidate_edges=len(obs.candidate_ids)))
        times=[]
        for _ in range(20):
            t=time.perf_counter()
            q=quality(data['pre_sinr'])
            for before,after,y in zip(data['before'],data['after'],q):resolve_topk(before,after,y,10)
            times.append(time.perf_counter()-t)
        rows.append(dict(checkpoint=ck,method='sinr_quality_plus_topk',median_seconds=float(np.median(times)),
            minimum_seconds=min(times),env_batch=len(data['before']),uav_groups=obs.groups,candidate_edges=len(obs.candidate_ids)))
    check_frozen(actor,meta)
atomic_json(root/'audit/inference_benchmark.json',dict(device=a.device,repetitions_diffusion=5,
    repetitions_rule=20,rows=rows,note='Only inference/rule computation; no physical step, I/O or preprocessing costs. Concurrent workload can influence timing.'))
print(rows,flush=True)
