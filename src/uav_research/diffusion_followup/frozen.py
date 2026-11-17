"""A3 frozen execution swap and full-snapshot one-step interventions."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ..runtime import resolve_output,atomic_json,atomic_save,isolated_rng,seed_all
from ..diffusion_dppo.config import from_dict
from ..diffusion_dppo.storage import load,configure
from ..diffusion_dppo.learner import DiffusionLearner
from ..diffusion_dppo.environment import ServiceEnvironment,ServiceVector,resolve_topk
from ..diffusion_dppo.evaluation import evaluate
from ..system_ablation.report import episode_metrics,write_csv
from ..topk_diagnostic import file_hash


def protocol_inputs(env,action):
    before=np.zeros_like(env.C,dtype=bool);scores=np.zeros_like(env.Y,dtype=np.float64)
    for i in range(env.cfg.num_uavs):
        lo,hi=action.offsets[i:i+2];ids=action.ue_ids[lo:hi]
        before[i,ids]=True;scores[i,ids]=action.scores[lo:hi]
    rule=np.exp(-np.logaddexp(0.,-(env.Y-env.cfg.sinr_threshold_db)/10.))
    return before,scores,rule,env.owner().copy()


def protocol_outcomes(env,prepared):
    before,learned,rule,previous=prepared
    after=env.sensed&(env.Y>=env.cfg.sinr_threshold_db)&(env.B.min(1)[:,None]<=env.cfg.num_uavs)
    left,lp,_=resolve_topk(before,after,learned,env.cfg.capacity)
    right,rp,_=resolve_topk(before,after,rule,env.cfg.capacity)
    np.testing.assert_array_equal(left,env.C)
    union=(lp|rp).sum();both=(lp&rp).sum()
    lo=np.where(left.any(0),left.argmax(0),-1);ro=np.where(right.any(0),right.argmax(0),-1)
    row=dict(proposal_jaccard=float(both/union) if union else 1.,
        proposal_changed_fraction=float((lp!=rp).sum()/max(1,union)),
        owner_changed_fraction=float((lo!=ro).mean()),candidate_edges=int(before.sum()),
        after_feasible_edges=int(after.sum()),after_feasible_users=int(after.any(0).sum()))
    for name,c,owner in [('learned',left,lo),('sinr',right,ro)]:
        row[name+'_coverage']=float(c.sum()/env.cfg.num_users)
        row[name+'_throughput_bps']=float((c*env.rates).sum())
        row[name+'_direct_switches']=int(((owner>=0)&(previous>=0)&(owner!=previous)).sum())
        row[name+'_service_interruptions']=int(((previous>=0)&(owner<0)).sum())
    return row


def swap_config(cfg):
    changed=copy.deepcopy(cfg)
    changed.diffusion.ablation.update(run_id='F2b',association='sinr_topk')
    return changed.validate()


def intervene(snapshot,action,learned_cfg,rule_cfg):
    """Only explicitly change protocol metadata; all physical/RNG/history stay exact."""
    environments=[];outcomes=[]
    for cfg in (learned_cfg,rule_cfg):
        env=ServiceEnvironment(cfg.env)
        saved=copy.deepcopy(snapshot)
        saved['service_ablation']=copy.deepcopy(cfg.env._service_ablation)
        env.restore(saved)
        result=env.step(copy.deepcopy(action))
        environments.append(env);outcomes.append(result[-1])
    a,b=environments
    np.testing.assert_array_equal(a.uav_xy,b.uav_xy)
    np.testing.assert_array_equal(a.user_xy,b.user_xy)
    np.testing.assert_array_equal(a.history,b.history)
    for key in ('A','B','Y','sensed'):np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-12,atol=1e-12)
    union=(a.proposals|b.proposals).sum();intersection=(a.proposals&b.proposals).sum()
    row=dict(proposal_jaccard=float(intersection/union) if union else 1.,
        proposal_changed_fraction=float((a.proposals!=b.proposals).sum()/max(1,union)),
        owner_changed_fraction=float((a.owner()!=b.owner()).mean()),
        available_edges=int((a.sensed&(a.Y>=a.cfg.sinr_threshold_db)&(a.B.min(1)[:,None]<=a.cfg.num_uavs)).sum()),
        physical_positions_equal=True)
    for key in ('coverage','throughput_bps','reward','direct_switches','service_interruptions',
                'proposal_count','proposal_acceptance','invalidated_proposals',
                'discovered_users','sensing_rejected_users','downlink_rejected_users','backhaul_rejected_users'):
        row['learned_'+key]=float(outcomes[0][key]);row['sinr_'+key]=float(outcomes[1][key])
        row['delta_'+key]=row['sinr_'+key]-row['learned_'+key]
    arrays=dict(learned_proposals=a.proposals.copy(),sinr_proposals=b.proposals.copy(),
                learned_association=a.C.copy(),sinr_association=b.C.copy())
    return row,arrays


@torch.no_grad()
def replay_interventions(learner,cfg,output,seeds,action_seed=880001,times=(0,150,200,399)):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    changed=swap_config(cfg);rows=[];protocol_rows=[]
    with isolated_rng():
        seed_all(action_seed)
        for lo in range(0,len(seeds),cfg.run.num_envs):
            chosen=seeds[lo:lo+cfg.run.num_envs]
            vector=ServiceVector(cfg.env,len(chosen),'batch')
            try:
                obs,_,_=vector.reset(chosen)
                for t in range(cfg.env.horizon):
                    action,chain=learner.actor.sample(obs)
                    items=action.split(cfg.env.num_uavs)
                    prepared=[protocol_inputs(e,item) for e,item in zip(vector.envs,items)]
                    if t in times:
                        # Same sampled chain used by real teacher and both shadow arms.
                        for i,(env,item) in enumerate(zip(vector.envs,items)):
                            snapshot=env.snapshot()
                            row,arrays=intervene(snapshot,item,cfg,changed)
                            row.update(scene_seed=chosen[i],t=t,action_seed=action_seed)
                            rows.append(row)
                            atomic_save(output/f'scene{chosen[i]}_t{t}.pt',dict(snapshot=snapshot,action=item,
                                chain=chain.take(np.arange(i*cfg.env.num_uavs,(i+1)*cfg.env.num_uavs)),
                                intervention='ONLY scoring source changed',metrics=row,**arrays))
                    obs,_,_,_,_,_=vector.step(action)
                    for scene,e,p in zip(chosen,vector.envs,prepared):
                        r=protocol_outcomes(e,p);r.update(scene_seed=scene,t=t,action_seed=action_seed)
                        protocol_rows.append(r)
            finally:vector.close()
    write_csv(output/'one_step_counterfactual.csv',rows)
    write_csv(output/'step_protocol_diagnostics.csv',protocol_rows)
    return rows


def frozen_swap(output,device='cpu',seeds=None,action_seeds=(880001,880002,880003),times=(0,150,200,399)):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=resolve_output('runs/system_ablation_1m/A3_seed0')
    rows=[];sources=[]
    for kind in ('step0','best','last'):
        checkpoint=old/(kind+'.pt');sha=file_hash(checkpoint)
        data=load(checkpoint,kinds={'dppo_model'})
        cfg=from_dict(data['config']);cfg.run.device=device;configure(cfg)
        learner=DiffusionLearner(cfg,device);learner.load_state_dict(data['learner']);learner.actor.eval()
        chosen=cfg.run.validation_seeds if seeds is None else list(seeds)
        for action_seed in action_seeds:
            for mode,env_cfg in [('learned',cfg),('sinr',swap_config(cfg))]:
                print(f'frozen A3 checkpoint={kind} scoring={mode} action_seed={action_seed}',flush=True)
                target=output/mode/kind/str(action_seed)
                cache=resolve_output(f'analysis/diffusion_gaussian_ablation_1m/A3/validation/{kind}/{action_seed}')
                if mode=='learned' and seeds is None and (cache/'complete.json').exists():
                    p=json.loads((cache/'provenance.json').read_text())
                    if p['sha256']!=sha or p['scene_seeds']!=chosen:raise ValueError('Frozen cache source mismatch')
                    target.mkdir(parents=True,exist_ok=True)
                    atomic_json(target/'source_reference.json',dict(path=str(cache),sha256=sha))
                    actual=cache
                else:
                    actual=target
                    if not (target/'complete.json').exists():
                        evaluate(env_cfg,learner.actor,chosen,target,critic=learner,action_seed=action_seed,traces=2)
                        atomic_json(target/'complete.json',dict(sha256=sha,action_seed=action_seed,scenes=chosen))
                    stamp=json.loads((target/'complete.json').read_text())
                    if stamp['sha256']!=sha or stamp['scenes']!=chosen or stamp['action_seed']!=action_seed:
                        raise ValueError('Frozen replay changed source')
                with np.load(actual/'step_metrics.npz',allow_pickle=False) as z:
                    points=episode_metrics(z['metrics'],z['names'].tolist(),z['scene_seeds'])
                rows.extend(dict(checkpoint=kind,scoring_source=mode,action_seed=action_seed,**r) for r in points)
                sources.append(dict(checkpoint=kind,scoring_source=mode,path=str(actual),sha256=sha,
                                    env_steps=data['env_steps'],action_seed=action_seed))
        interventions=output/'one_step'/kind
        if (interventions/'complete.json').exists():
            saved=json.loads((interventions/'complete.json').read_text())
            if (saved['sha256']!=sha or saved['scenes']!=chosen or saved['times']!=list(times)
                    or saved['action_seed']!=action_seeds[0]):raise ValueError('One-step replay source changed')
        if not (interventions/'complete.json').exists() or not (interventions/'step_protocol_diagnostics.csv').exists():
            replay_interventions(learner,cfg,interventions,chosen,action_seeds[0],times)
            atomic_json(interventions/'complete.json',dict(sha256=sha,scenes=chosen,times=times,action_seed=action_seeds[0]))
        if file_hash(checkpoint)!=sha:raise ValueError('Source model changed')
    write_csv(output/'per_scene_frozen.csv',rows)
    paired=[]
    for kind in ('step0','best','last'):
        scenes=sorted({r['scene_seed'] for r in rows if r['checkpoint']==kind})
        draws=np.random.default_rng(950001).integers(len(scenes),size=(2000,len(scenes)))
        for metric in ('last100_coverage','final_coverage','last100_throughput_mbps','last100_coverage_std_pp',
                       'last100_drop_events_10pp','last100_direct_switches','last100_service_interruptions'):
            x=[]
            for scene in scenes:
                values={mode:np.mean([r[metric] for r in rows if r['checkpoint']==kind and r['scene_seed']==scene
                        and r['scoring_source']==mode]) for mode in ('learned','sinr')}
                x.append(values['sinr']-values['learned'])
            x=np.array(x);low,high=np.quantile(x[draws].mean(1),[.025,.975])
            paired.append(dict(checkpoint=kind,comparison='frozen SINR - learned',metric=metric,
                difference=float(x.mean()),low=float(low),high=float(high)))
    write_csv(output/'frozen_delta.csv',paired)
    atomic_json(output/'sources.json',sources)
    full=seeds is None and len(action_seeds)==3
    atomic_json(output/'complete.json',dict(full=full,scenes=chosen,action_seeds=action_seeds,
        intervention_times=times,closed_loop='After first different association, states and future actions may diverge',
        service_event_units='last100_direct_switches and interruptions are counts per time step; multiply by100 for per-episode window totals'))
    # Saved-data figures; no new actor calls and no change to source trajectories.
    import matplotlib.pyplot as plt
    from ..plotting import style,save_figure
    style()
    for kind in ('step0','best','last'):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for mode,color,ls in [('learned','#009E73','-'),('sinr','#0072B2','--')]:
            values=[]
            for s in sources:
                if s['checkpoint']==kind and s['scoring_source']==mode:
                    with np.load(Path(s['path'])/'step_metrics.npz') as z:
                        values.append((z['metrics'],z['names'].tolist()))
            for ax,key,scale in zip(axes,('coverage','throughput_bps'),(100,1e-6)):
                curve=np.mean([x[:,:,names.index(key)].mean(0) for x,names in values],0)*scale
                ax.plot(np.arange(1,len(curve)+1),curve,c=color,ls=ls,label=mode)
                ax.set(xlabel='Episode step',ylabel='Coverage (%)' if key=='coverage' else 'Throughput (Mbps)');ax.legend()
        save_figure(fig,output/'figures',kind+'_frozen_scoring_swap')
    return paired
