"""Saved-state geometric risk, paired fork contrasts and causal topology cases."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from uav_research.diffusion_dppo.config import from_dict
from uav_research.diffusion_dppo.diagnosis import read_csv, write_csv, scene_bootstrap, sampled_backhaul_hops
from uav_research.diffusion_dppo.environment import ServiceEnvironment
from uav_research.diffusion_dppo.evaluation import frame
from uav_research.plotting import style, save_figure

ROOT=Path('analysis/diffusion_gaussian_diagnosis_20261110')
SOURCE=Path('analysis/hop_scaling_joint_service_10m_seed0')

def read_states(kind):
    states={}
    for group in ['diffusion','gaussian']:
        data=torch.load(ROOT/'states'/f'{kind}_{group}.pt',weights_only=False,map_location='cpu')
        cfg=from_dict(data['config'])
        states.update({s['state_id']:s for s in data['states']})
    return cfg,states

def fork_contrasts():
    rows=[]
    for kind in ['best','last']:
        with np.load(ROOT/'exp1_motion_vs_scoring'/f'forks_{kind}.npz') as z:
            values=z['values']; names=z['names'].tolist();combos=z['combinations'].tolist()
            ids=[json.loads(x) for x in z['identities_json']]
        for left,right in [('G/D','D/D'),('G/G','D/D'),('D/G','D/D'),('G/D','G/G')]:
            for horizon in [10,25,50]:
                d=(values[:,combos.index(left),:horizon,names.index('coverage')]-values[:,combos.index(right),:horizon,names.index('coverage')]).mean(1)
                scenes={}
                for identity,value in zip(ids,d):scenes.setdefault(identity['scene'],[]).append(value)
                rows.append(dict(kind=kind,left=left,right=right,horizon=horizon,**scene_bootstrap([np.mean(v) for v in scenes.values()])))
    write_csv(ROOT/'exp1_motion_vs_scoring/fork_paired_contrasts.csv',rows)

def risk():
    rows=[]
    for kind in ['best','last']:
        cfg,states=read_states(kind)
        for policy in ['D0','Db','Dl','Gb','Gl']:
            with np.load(ROOT/'exp2_action_distribution'/f'conditional_samples_{kind}_{policy}.npz') as z:
                for index,sid in enumerate(z['state_ids']):
                    item=states[str(sid)];snap=item['snapshot']
                    env=ServiceEnvironment(cfg.env);env.restore(snap)
                    hops=sampled_backhaul_hops(env.uav_xy,z['actual'][index],cfg.env)
                    was=env.B.min(1)<=cfg.env.num_uavs
                    lost=(hops>cfg.env.num_uavs)&was[None]
                    counts=env.C.sum(1)
                    rows.append(dict(kind=kind,policy=policy,state_id=str(sid),scene=item['scene'],is_fixed=int(item['is_fixed']),
                        coverage_before=env.C.sum()/cfg.env.num_users,p_any_new_disconnect=float(lost.any(1).mean()),
                        expected_new_disconnected=float(lost.sum(1).mean()),incumbent_users_exposed=float((lost*counts[None]).sum(1).mean())))
            print('risk',kind,policy,flush=True)
    write_csv(ROOT/'exp2_action_distribution/geometric_backhaul_risk.csv',rows)
    summary=[]
    for kind in ['best','last']:
        for policy in ['D0','Db','Dl','Gb','Gl']:
            for metric in ['p_any_new_disconnect','expected_new_disconnected','incumbent_users_exposed']:
                scenes={}
                for r in rows:
                    if r['kind']==kind and r['policy']==policy and r['is_fixed']:scenes.setdefault(r['scene'],[]).append(r[metric])
                summary.append(dict(kind=kind,policy=policy,metric=metric,**scene_bootstrap([np.mean(x) for x in scenes.values()])))
    write_csv(ROOT/'exp2_action_distribution/geometric_risk_summary.csv',summary)

def cases():
    import matplotlib.pyplot as plt
    rows=[]
    cases=[('best','diffusion',10012,40,3),('best','gaussian',10025,321,13),('last','diffusion',10028,390,16),('last','gaussian',10022,217,15)]
    fig,axes=plt.subplots(4,3,figsize=(13,16))
    for line,(kind,group,scene,t,held) in enumerate(cases):
        cfg,states=read_states(kind);sid=f'{kind}:{group}:{scene}:{t}'
        data=torch.load(ROOT/'states'/f'factual_{kind}_{group}.pt',weights_only=False,map_location='cpu')
        f=data[sid];env=ServiceEnvironment(cfg.env);env.restore(states[sid]['snapshot'])
        before=frame(env);action=copy.deepcopy(f['factual_action']);env.step(action);after=frame(env)
        np.testing.assert_array_equal(after['C'],f['after']['C'])
        env.restore(states[sid]['snapshot']);action.movement[held]=0;env.step(action);hold=frame(env)
        bs=np.asarray(cfg.env.bs_positions);p0=before['uav_xy'];p1=after['uav_xy']
        oldmin=np.sqrt(((p0[held]-bs[:,:2])**2).sum(1)+(cfg.env.altitude-bs[:,2])**2).min()
        newmin=np.sqrt(((p1[held]-bs[:,:2])**2).sum(1)+(cfg.env.altitude-bs[:,2])**2).min()
        rows.append(dict(kind=kind,group=group,scene=scene,action_step=t+1,held_uav_zero_based=held,
            before_coverage=float(before['C'].sum()/150),after_coverage=float(after['C'].sum()/150),hold_coverage=float(hold['C'].sum()/150),
            movement_m=float(np.linalg.norm(p1[held]-p0[held])),nearest_bs_before_m=float(oldmin),nearest_bs_after_m=float(newmin)))
        for col,(label,fr) in enumerate([('Before',before),('Factual action',after),('Hold one UAV',hold)]):
            ax=axes[line,col];pos=fr['uav_xy'];users=fr['user_xy'];online=fr['B'].min(1)<=18
            served=fr['C'].any(0)
            ax.scatter(users[:,0],users[:,1],c=np.where(served,'#0072B2','#bbbbbb'),s=8,alpha=.5)
            for i in range(18):
                for j in range(i):
                    if fr['A'][i,j]:ax.plot(pos[[i,j],0],pos[[i,j],1],color='#999999',lw=.5)
                for b in bs:
                    if np.sqrt(((pos[i]-b[:2])**2).sum()+(cfg.env.altitude-b[2])**2)<=cfg.env.ug_range:
                        ax.plot([pos[i,0],b[0]],[pos[i,1],b[1]],color='#0072B2',lw=.8,ls='--')
                ax.text(pos[i,0]+20,pos[i,1]+20,str(i),fontsize=6)
            ax.scatter(pos[:,0],pos[:,1],c=np.where(online,'#009E73','#555555'),s=35,marker='^')
            ax.scatter(*pos[held],s=130,facecolors='none',edgecolors='#000000',lw=2)
            ax.scatter(bs[:,0],bs[:,1],marker='s',c='#000000',s=45)
            ax.set(xlim=(-80,3580),ylim=(-80,3580),aspect='equal',title=f'{group} {kind} s{scene} t{t+1}: {label}\ncoverage {fr["C"].sum()/150:.1%}; offline {(~online).sum()}')
    save_figure(fig,ROOT/'exp1_motion_vs_scoring','drop_case_studies')
    write_csv(ROOT/'exp1_motion_vs_scoring/causal_case_summary.csv',rows)

def noise():
    rows=[]
    for kind in ['best','last']:
        for group in ['diffusion','gaussian']:
            path=ROOT/'exp2_action_distribution'/f'noise_{kind}_{group}'
            if not (path/'episodes.csv').exists():continue
            a=read_csv(path/'episodes.csv');b=read_csv(SOURCE/kind/group/'episodes.csv')
            assert [x['scene_seed'] for x in a]==[x['scene_seed'] for x in b]
            for metric in ['coverage','last100_coverage','return','actual_speed_mean','disconnection_ratio']:
                rows.append(dict(kind=kind,group=group,metric=metric,original=float(np.mean([float(x[metric]) for x in b])),
                    intervention=float(np.mean([float(x[metric]) for x in a])),**scene_bootstrap([float(x[metric])-float(y[metric]) for x,y in zip(a,b)])))
    write_csv(ROOT/'exp2_action_distribution/noise_paired_comparison.csv',rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fork','risk','cases','noise']);args=p.parse_args()
    torch.set_num_threads(2);style()
    {'fork':fork_contrasts,'risk':risk,'cases':cases,'noise':noise}[args.stage]()
