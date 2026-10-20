"""Read-only policy diagnostics. No training or baseline policies."""
from pathlib import Path
import csv,json
import numpy as np
import torch
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.runtime import configure_runtime
from uav_research.vector import VectorEnvironment
from uav_research.actions import DISCRETE_DIRECTIONS
from uav_research.kernels import shortest_hops, associate
from uav_research.plotting import style,save_figure,plt,COLORS

RUN=Path('runs/wsl_discrete_mappo_paper_scene_250k_seed0')
OUT=Path('analysis/mappo_discrete_250k_seed0/audit')
OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(RUN/'config.yaml')
device=configure_runtime(cfg)

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(env):
    e=env.cfg;p=env.uav_xy
    on_edge=np.any((p<=1e-5)|(p>=[e.width-1e-5,e.height-1e-5]),axis=1)
    corner=np.all((p<=30)|(p>=[e.width-30,e.height-30]),axis=1)
    top_right=np.all(p>=[e.width-30,e.height-30],axis=1)
    return dict(coverage=env.info['coverage'],reward=env.info['reward'],
        discovered=env.info['discovery_fraction'],disconnected=env.info['disconnected_uavs'],
        edge=float(on_edge.mean()),corner=float(corner.mean()),top_right=float(top_right.mean()),
        backhaul_rejected=env.info['backhaul_rejected_users'],
        sensing_rejected=env.info['sensing_rejected_users'],
        downlink_rejected=env.info['downlink_rejected_users'],
        capacity_rejected=env.info['capacity_rejected_users'],
        clip=env.info['boundary_clip_fraction'])

def ranges(env):
    e=env.cfg;u=e.num_uavs
    squared=np.sum((env.uav_xy[:,None]-np.asarray(e.bs_positions)[None,:,:2])**2,axis=-1)+e.altitude**2
    pair=np.sum((env.uav_xy[:,None]-env.uav_xy[None,:])**2,axis=-1)
    cand=np.where(env.sensed,env.Y,-np.inf)
    variants=[('current',650,1000),('UG1200',650,1200),('UG1500',650,1500),
              ('UG2000',650,2000),('UU780',780,1000),('UU1000',1000,1000),
              ('UU1000_UG1500',1000,1500),('ideal_backhaul',None,None)]
    rows=[]
    for name,uu,ug in variants:
        h=(np.ones((u,e.num_bs),np.int64) if uu is None else
           shortest_hops(pair<=uu*uu,squared<=ug*ug,np.ones(u,bool)))
        c,n,_=associate(cand,h,e.sinr_threshold_db,e.capacity)
        rows.append(dict(variant=name,coverage=float(n.sum()/e.num_users),
                         reachable=float((h.min(1)<=u).mean())))
    # No physical changes: discovery/downlink/capacity unchanged, re-association allowed.
    return rows

all_curves=[];all_episodes=[];all_ranges=[];policies=[]
for checkpoint in ['best','last']:
    pcfg, policy=policy_from_checkpoint(RUN/f'{checkpoint}.pt',device)
    for mode, action_seed in [('argmax',0),('sample',880001),('sample',880002)]:
        torch.manual_seed(action_seed)
        histories=[];actions_hist=[];probs_hist=[];values_hist=[];reward_hist=[]
        for start in range(0,len(pcfg.run.validation_seeds),16):
            seeds=pcfg.run.validation_seeds[start:start+16]
            with VectorEnvironment(pcfg.env,len(seeds),'batch') as vector:
                obs,state,_=vector.reset(seeds)
                logs=[[] for _ in seeds];ps=[];acts=[];vs=[];rs=[]
                for t in range(pcfg.env.horizon+1):
                    for i,env in enumerate(vector.envs):
                        item=dict(checkpoint=checkpoint,mode=mode,action_seed=action_seed,
                                  scene_seed=seeds[i],step=t,**stats(env))
                        logs[i].append(item)
                        if mode=='argmax' and t%20==0:
                            for rr in ranges(env):
                                all_ranges.append(dict(checkpoint=checkpoint,scene_seed=seeds[i],step=t,**rr))
                    if t==pcfg.env.horizon:break
                    with torch.no_grad():
                        dist=policy.actor.distribution(obs.to_torch(device))
                        p=dist.probs.cpu().numpy()
                        a=dist.probs.argmax(-1) if mode=='argmax' else dist.sample()
                        a=a.cpu().numpy()
                        vs.append(policy.values(state).cpu().numpy())
                    ps.append(p);acts.append(a)
                    obs,state,r,_,_,_=vector.step(a);rs.append(r)
                ps=np.stack(ps);acts=np.stack(acts);vs=np.stack(vs);rs=np.stack(rs)
                g=np.zeros(len(seeds));returns=np.zeros_like(rs)
                for t in reversed(range(len(rs))):
                    g=rs[t]+pcfg.algo.discount*g;returns[t]=g
                ev=1-np.var(returns-vs)/np.var(returns)
                for i,seed in enumerate(seeds):
                    h=logs[i];episode=dict(checkpoint=checkpoint,mode=mode,action_seed=action_seed,scene_seed=seed)
                    for k in stats(vector.envs[i]):
                        episode['initial_'+k]=h[0][k]
                        episode['mean_'+k]=np.mean([row[k] for row in h[1:]])
                        episode['final_'+k]=h[-1][k]
                    episode['return']=float(rs[:,i].sum())
                    episode['max_prob']=float(ps[:,i].max(-1).mean())
                    episode['top2_gap']=float(np.diff(np.sort(ps[:,i],axis=-1)[...,-2:],axis=-1).mean())
                    episode['entropy']=float(-(ps[:,i]*np.log(np.maximum(ps[:,i],1e-30))).sum(-1).mean())
                    expected=ps[:,i]@DISCRETE_DIRECTIONS
                    episode['mean_expected_speed']=float(np.linalg.norm(expected,axis=-1).mean()*30)
                    episode['value_mc_rmse']=float(np.sqrt(np.mean((vs[:,i]-returns[:,i])**2)))
                    for k in range(9):
                        episode[f'action_{k}']=float((acts[:,i]==k).mean())
                        episode[f'prob_{k}']=float(ps[:,i,:,k].mean())
                    all_episodes.append(episode)
                # Curves averaged later over all scenes; boundary includes t=0.
                for t in range(pcfg.env.horizon+1):
                    all_curves.append(dict(checkpoint=checkpoint,mode=mode,action_seed=action_seed,batch=start,step=t,
                        **{k:np.mean([h[t][k] for h in logs]) for k in stats(vector.envs[0])}))
        subset=[r for r in all_episodes if r['checkpoint']==checkpoint and r['mode']==mode and r['action_seed']==action_seed]
        print(checkpoint,mode,action_seed,{k:round(float(np.mean([r[k] for r in subset])),5) for k in
            ['initial_coverage','mean_coverage','final_coverage','final_edge','final_top_right','max_prob','entropy','mean_expected_speed']},flush=True)
write_csv('episodes.csv',all_episodes);write_csv('curves.csv',all_curves);write_csv('range_counterfactual.csv',all_ranges)
summary={}
for checkpoint in ['best','last']:
    for mode in ['argmax','sample']:
        rows=[r for r in all_episodes if r['checkpoint']==checkpoint and r['mode']==mode]
        summary[f'{checkpoint}_{mode}']={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['checkpoint','mode','action_seed','scene_seed']}
range_summary=[]
for checkpoint in ['best','last']:
    for variant in sorted(set(r['variant'] for r in all_ranges)):
        rows=[r for r in all_ranges if r['checkpoint']==checkpoint and r['variant']==variant]
        range_summary.append(dict(checkpoint=checkpoint,variant=variant,
            initial_coverage=float(np.mean([r['coverage'] for r in rows if r['step']==0])),
            sampled_time_coverage=float(np.mean([r['coverage'] for r in rows if r['step']>0])),
            final_coverage=float(np.mean([r['coverage'] for r in rows if r['step']==400]))))
write_csv('range_summary.csv',range_summary)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
style();fig,axs=plt.subplots(1,3,figsize=(12,3.5))
for checkpoint,style_ in [('best','-'),('last','--')]:
    for mode,color in [('argmax',COLORS[1]),('sample',COLORS[0])]:
        rr=[r for r in all_curves if r['checkpoint']==checkpoint and r['mode']==mode]
        for ax,key in zip(axs,['coverage','edge','disconnected']):
            vals=[np.mean([r[key] for r in rr if r['step']==t]) for t in range(401)]
            ax.plot(range(401),vals,color=color,ls=style_,label=f'{checkpoint} / {mode}')
for ax,y in zip(axs,['Service coverage','UAV fraction exactly on boundary','Disconnected UAV count']):
    ax.set(xlabel='Episode step',ylabel=y);ax.legend(fontsize=7)
save_figure(fig,OUT,'policy_execution')
print('Saved audit:',OUT,flush=True)
