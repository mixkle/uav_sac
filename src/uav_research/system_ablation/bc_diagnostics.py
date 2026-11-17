"""Fixed-observation imitation checks, separate from BC selection and RL."""
import numpy as np
import torch
from ..diffusion_dppo.observations import offsets
from .settings import options


def diagnose(actor,dataset,cfg,limit=1024):
    ids=np.linspace(0,len(dataset)-1,min(limit,len(dataset)),dtype=np.int64)
    squared,count,overlap,top_count=0.,0,0,0
    with torch.no_grad():
        for lo in range(0,len(ids),256):
            obs,m,s=dataset.batch(ids[lo:lo+256])
            _,chain=actor.sample(obs,deterministic=True)
            predicted=chain.movement[:,-1]
            squared+=float(np.square(predicted-m).sum());count+=m.size
            if options(cfg)['score_actions']:
                starts=offsets(obs.candidate_lengths)
                for begin,end in zip(starts[:-1],starts[1:]):
                    k=min(cfg.env.capacity,end-begin)
                    if not k:continue
                    teacher=np.argsort(-s[begin:end],kind='stable')[:k]
                    student=np.argsort(-chain.scores[begin:end,-1],kind='stable')[:k]
                    overlap+=len(set(teacher)&set(student));top_count+=k
    return dict(samples=len(ids),motion_latent_rmse_against_teacher_sample=float(np.sqrt(squared/max(1,count))),
        topk_overlap=float(overlap/top_count) if top_count else None,
        mode='deterministic Gaussian mean / zero-noise DDPM diagnostic; not a final distribution mean',
        note='Supervised labels are teacher SAMPLES; single-action error is not distributional imitation accuracy')
