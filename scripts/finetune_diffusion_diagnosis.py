"""Controlled NEW fine-tuning runs from a frozen best actor AND critic.

Missing environment/RNG history is not invented as a seamless continuation.
Both arms reset Adam and counters, retain the same best value normalizers,
and start fresh, identically seeded environments. Production runner unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from uav_research.config import save_config
from uav_research.runtime import seed_all, rng_state, atomic_json, atomic_save
from uav_research.topk_diagnostic import file_hash
from uav_research.diffusion_dppo.config import from_dict
from uav_research.diffusion_dppo.environment import ServiceVector
from uav_research.diffusion_dppo.evaluation import evaluate
from uav_research.diffusion_dppo.learner import DiffusionLearner
from uav_research.diffusion_dppo.runner import train_online
from uav_research.diffusion_dppo.storage import load, artifact, save_online_full, configure
from uav_research.diffusion_dppo.diagnosis import write_csv


def tensor_hash(modules):
    value=hashlib.sha256()
    for name, module in sorted(modules.items()):
        for key, tensor in sorted(module.state_dict().items()):
            value.update((name+'/'+key).encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['control','conservative'],required=True)
    p.add_argument('--source',type=Path,default=Path('analysis/hop_scaling_joint_service_10m_seed0/snapshots/diffusion/best.pt'))
    p.add_argument('--output',type=Path,default=Path('analysis/diffusion_gaussian_diagnosis_20261110/exp4_conservative_finetuning'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--budget',type=int,default=200000)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    gate=args.output/'gate_decision.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('run_screening'):
        raise ValueError('Experiments1–3 must record an evidence-based screening decision first')
    source=load(args.source,kinds={'dppo_model'})
    source_hash=file_hash(args.source)
    cfg=from_dict(source['config'])
    cfg.run.total_steps=args.budget
    cfg.run.device=args.device
    cfg.run.cpu_threads=cfg.run.numba_threads=4
    cfg.run.eval_interval=50000
    cfg.run.checkpoint_minutes=10
    cfg.run.tensorboard=False
    cfg.diffusion.actor_update_devices=[]
    if args.group=='conservative':
        cfg.algo.actor_lr=3e-5
        cfg.algo.ppo_epochs=4
        cfg.algo.ppo_clip=.05
    cfg.validate()
    output=args.output/args.group
    output.mkdir(parents=True,exist_ok=True)
    config_path=args.output/'configs'/(args.group+'.yaml')
    config_path.parent.mkdir(parents=True,exist_ok=True)
    save_config(cfg,config_path)
    if args.resume:
        result=train_online(cfg,output,resume=args.resume)
        assert file_hash(args.source)==source_hash
        print(json.dumps(result),flush=True)
        return
    if (output/'initialization.json').exists():
        raise FileExistsError('Existing initialization: choose explicit --resume; never overwrite')
    device=configure(cfg);seed_all(cfg.run.seed)
    learner=DiffusionLearner(cfg,device)
    copied=['actor','value','normalizer','average_normalizer','horizon_factor']
    for key in copied:
        getattr(learner,key).load_state_dict(source['learner'][key])
    assert len(learner.actor_opt.state)==0 and len(learner.value_opt.state)==0
    learner.updates=learner.optimizer_steps=0
    state_hash=tensor_hash({k:getattr(learner,k) for k in copied})
    # Isolated evaluation restores all training RNGs; same best actor validation.
    initial=evaluate(cfg,learner.actor,cfg.run.validation_seeds,output/'initial_evaluation',
                     critic=learner,moment_repeats=0)
    expected=270.9868907535582
    if args.budget==200000 and not np.isclose(initial['summary']['return'],expected,atol=1e-6,rtol=0):
        raise AssertionError('Best initialization failed its original validation reproduction')
    workers=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg.run.seed,101]).spawn(cfg.run.num_envs)]
    vector=ServiceVector(cfg.env,cfg.run.num_envs,'batch')
    obs,state,_=vector.reset(workers)
    provenance=dict(mode='new_finetune_from_best',source_checkpoint=str(args.source),
        source_sha256=source_hash,source_original_env_steps=source['env_steps'],
        actor_and_critic='same best; both copied with value normalizers',
        optimizers='both Adam reset; source Adam deliberately not inherited',
        environment_rng='new same seed0 worker streams, not original best environment state',
        initialized_modules_sha256=state_hash,extra_budget=args.budget,group=args.group)
    model=artifact(cfg,'model',env_steps=0,learner=learner.state_dict(),provenance=provenance)
    for name in ['step0.pt','best.pt','last.pt']:
        atomic_save(output/name,model)
    full=artifact(cfg,'full',env_steps=0,learner=learner.state_dict(),
        environments=vector.snapshot(),observation=obs.copy(),state=state.copy(),
        rng=rng_state(),best_score=initial['summary']['return'],best_step=0,next_eval=50000,
        num_envs=cfg.run.num_envs,elapsed_seconds=0.,provenance=provenance,
        phase_times={'evaluation_seconds':0.,'io_seconds':0.},best_artifact=model)
    path=save_online_full(output/'origin',cfg,full)
    vector.close()
    row=dict(env_steps=0,elapsed_seconds=0.,**{k:initial['summary'][k] for k in
        ['return','coverage','throughput_bps','last100_coverage','final_coverage']},
        task_return=initial['summary']['return'],best_score=initial['summary']['return'],best_step=0)
    fields=['env_steps','elapsed_seconds','return','task_return','coverage','throughput_bps',
            'last100_coverage','final_coverage','best_score','best_step']
    write_csv(output/'validation.csv',[{k:row[k] for k in fields}])
    atomic_json(output/'initialization.json',dict(provenance,bootstrap_checkpoint=str(path),
        initial_return=initial['summary']['return'],initial_coverage=initial['summary']['coverage'],
        original_best_is_full_checkpoint=False,source_has_optimizer_state=True,
        source_has_environment_and_rng=False))
    if args.prepare_only:
        print('PREPARED',path,flush=True)
        return
    print('START_SCREENING',args.group,'budget',args.budget,'source',source['env_steps'],flush=True)
    result=train_online(cfg,output,resume=path)
    assert file_hash(args.source)==source_hash
    print('SCREENING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
