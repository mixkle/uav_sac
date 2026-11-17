"""S1/S2 frozen evaluation. Bounded replay, independent calibration, closed loops."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from .score_rule_diagnostic import (ScoreRuleEnvironment, rule_scores, quality,
    ranking_rows, competition_metrics, interruption_rows, survival_rows, chosen)
from .association_timing import sinr_scores, outcome
from .association_timing_experiment import ACTION_SEEDS, SCENES
from .environment import physical_links
from .diffusion_dppo.environment import resolve_topk
from .diffusion_dppo.config import from_dict
from .diffusion_dppo.storage import load, configure, ensure_budget, Progress
from .diffusion_dppo.policy import make_service_actor
from .diffusion_dppo.observations import stack, ServiceAction
from .diffusion_dppo.evaluation import frame
from .diffusion_followup.runtime import actor_digest
from .runtime import resolve_output, atomic_json, isolated_rng, seed_all
from .topk_diagnostic import atomic_npz, file_hash
from .system_ablation.report import episode_metrics, write_csv

DEFAULT_OUTPUT='analysis/score_rule_s1_s2_20261116'
REFERENCE='analysis/t1_t2_t3_post_motion_association'
CALIBRATION=list(range(740000,740016))
CONFIRMATION=list(range(750000,750032))
FIXED_STEPS=(50,100,150,200,250,300,325,350,375,400)
GRID=(0.,.05,.15,.30)
VERSION=1


def read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def source_actor(checkpoint,device):
    source=resolve_output(f'runs/system_ablation_1m/A3_seed0/{checkpoint}.pt')
    digest=file_hash(source); data=load(source,kinds={'dppo_model'})
    cfg=from_dict(data['config']);cfg.run.device=device
    cfg.run.cpu_threads=8;cfg.run.numba_threads=4;cfg.diffusion.actor_update_devices=[]
    configure(cfg)
    if cfg.diffusion.ablation['association']!='learned_topk':raise ValueError('Not original A3')
    actor=make_service_actor(cfg).to(device);actor.load_state_dict(data['learner']['actor'])
    actor.eval();actor.requires_grad_(False)
    return cfg,actor,dict(checkpoint=checkpoint,checkpoint_path=str(source),checkpoint_sha256=digest,
        actual_online_steps=data['env_steps'],actor_sha256=actor_digest(actor),optimizer_created=False,
        version=VERSION)


def check_frozen(actor,meta):
    assert actor_digest(actor)==meta['actor_sha256']
    assert file_hash(meta['checkpoint_path'])==meta['checkpoint_sha256']
    assert all(p.grad is None and not p.requires_grad for p in actor.parameters())


def numeric_info(e,result):
    return dict(result[-1],near_hover_fraction=float((np.linalg.norm(e.actual_displacement,axis=1)<1).mean()))


def validate_reward(e):
    expected=e.C.sum()/e.cfg.num_users+.1*(e.C*e.rates).sum()/(e.cfg.num_users*5e6)
    np.testing.assert_allclose(e.info['reward'],expected,rtol=0,atol=1e-12)


def finalize_episodes(array,keys,seeds):
    rows=episode_metrics(array,keys,seeds)
    sums=('direct_switches','service_interruptions','service_reconnections','link_retained',
          'previous_links','completed_service_duration_sum','completed_service_spells')
    extras=('idle_slots','full_uav_fraction','candidate_count_mean','valid_proposal_multi_ue_fraction',
            'served_ue_rate_mean_mbps','served_ue_rate_p05_mbps')
    for row,x in zip(rows,array):
        for n in (100,50):
            for key in sums:
                row[f'last{n}_{key}_window_sum']=float(x[-n:,keys.index(key)].sum())
                row[f'last{n}_{key}_per_step']=float(x[-n:,keys.index(key)].mean())
            row[f'last{n}_link_retention']=row[f'last{n}_link_retained_window_sum']/max(1,row[f'last{n}_previous_links_window_sum'])
            for key in extras:
                row[f'last{n}_{key}']=float(x[-n:,keys.index(key)].mean())
        for n in range(11):row[f'last100_load_{n}_fraction']=float(x[-100:,keys.index(f'load_{n}_fraction')].mean())
        ended=x[:,keys.index('completed_service_spells')].sum()
        row['completed_spell_mean']=float(x[:,keys.index('completed_service_duration_sum')].sum()/ended) if ended else None
    return rows


def append_spells(rows,e,previous,ages,scene,t):
    current=e.owner();ended=(previous>=0)&(current!=previous)&(ages>0)
    for m in np.flatnonzero(ended):
        rows.append(dict(scene_seed=scene,ue=int(m),end_step=t,duration=int(ages[m]),event=1,
                         initial_spell=int(ages[m]>=t-1)))


def locked(root):
    p=Path(root)/'configs/locked_rules.json'
    data=json.loads(p.read_text())
    payload=dict(data);digest=payload.pop('contract_sha256')
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==digest
    return data


@torch.no_grad()
def closed_case(checkpoint,rule,action_seed,output,device='cuda:0',split='development',smoke=False,
                calibration_accumulator=None):
    cfg,actor,meta=source_actor(checkpoint,device)
    lock=locked(output) if split!='calibration' else dict(beta=0.,load_weight=0.)
    scenes={'development':SCENES,'confirmation':CONFIRMATION,'calibration':CALIBRATION}[split]
    horizon=400
    if smoke:scenes=scenes[:4];horizon=8
    target=Path(output)/'s2/rollout'/split/checkpoint/rule/str(action_seed)
    contract=dict(meta,rule=rule,action_seed=action_seed,scenes=scenes,horizon=horizon,
                  beta=lock['beta'],load_weight=lock['load_weight'],split=split)
    if (target/'complete.json').exists():
        assert json.loads((target/'complete.json').read_text())==contract
        print('reuse',target,flush=True);return
    ensure_budget(cfg,target,estimated_bytes=100*2**20);target.mkdir(parents=True,exist_ok=True)
    metrics=[];keys=None;events=[];spells=[];user_hash=hashlib.sha256();progress=Progress(25)
    start=time.perf_counter();peak=0
    with isolated_rng():
        seed_all(action_seed)
        for begin in range(0,len(scenes),16):
            chosen_scenes=scenes[begin:begin+16]
            envs=[ScoreRuleEnvironment(cfg.env,rule,lock['beta'],lock['load_weight']) for _ in chosen_scenes]
            obs=stack([e.reset(s)[0] for e,s in zip(envs,chosen_scenes)])
            user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
            frames={i:[frame(e)] for i,e in enumerate(envs) if chosen_scenes[i] in (scenes[0],scenes[1])}
            batch=[];block=[]
            for t in range(horizon):
                action,chain=actor.sample(obs)
                assert np.isfinite(chain.logp).all()
                items=action.split(cfg.env.num_uavs)
                prepared=[e.prepare(a) for e,a in zip(envs,items)]
                links=physical_links(cfg.env,np.stack([e.uav_xy for e in envs]),np.stack([e.user_xy for e in envs]))
                results=[e.finish(p,tuple(x[i] for x in links)) for i,(e,p) in enumerate(zip(envs,prepared))]
                infos=[numeric_info(e,r) for e,r in zip(envs,results)]
                if keys is None:keys=sorted(k for k,v in infos[0].items() if np.isscalar(v))
                batch.append([[r[k] for k in keys] for r in infos])
                user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
                for i,e in enumerate(envs):
                    validate_reward(e)
                    append_spells(spells,e,prepared[i][2],e.pre_age,chosen_scenes[i],t+1)
                    if t>=300:
                        rows=interruption_rows(e.pre_c,e.C,prepared[i][0],e.diagnostic_after,e.proposals,
                            e.B,e.pre_y,e.Y,e.pre_age,cfg.env.capacity)
                        events.extend(dict(scene_seed=chosen_scenes[i],step=t+1,**r) for r in rows)
                    if i in frames:frames[i].append(frame(e))
                    if calibration_accumulator is not None and (t+1)%5==0:
                        calibration_accumulator(e,prepared[i][0],e.diagnostic_after,checkpoint,chosen_scenes[i],t+1)
                block.append([dict(uav_xy=e.uav_xy.copy(),user_xy=e.user_xy.copy(),association=np.packbits(e.C),
                    before=np.packbits(p[0]),after=np.packbits(e.diagnostic_after),proposals=np.packbits(e.proposals),
                    scores=e.last_scores.copy(),movement=a.movement.copy(),B=e.B.copy(),service_age=e.service_age.copy())
                    for e,p,a in zip(envs,prepared,items)])
                if len(block)==50 or t+1==horizon:
                    atomic_npz(target/'closed_loop'/f'b{begin}_t{t+1-len(block):03d}.npz',
                        **{k:np.stack([[e[k] for e in row] for row in block]) for k in block[0][0]},
                        scene_seeds=chosen_scenes,first_step=t+2-len(block),shape=[18,150])
                    block=[]
                obs=stack([r[0] for r in results]);peak=max(peak,psutil.Process().memory_info().rss)
                if peak>12*2**30:raise MemoryError('RSS exceeds 12 GiB')
                progress(f'{split}/{checkpoint}/{rule}/{action_seed} scenes={chosen_scenes[0]} t={t+1}/{horizon}')
            metrics.append(np.asarray(batch,float).transpose(1,0,2))
            for i,e in enumerate(envs):
                for m in np.flatnonzero(e.C.any(0)):
                    if e.service_age[m]>0:spells.append(dict(scene_seed=chosen_scenes[i],ue=int(m),end_step=horizon,
                        duration=int(e.service_age[m]),event=0,initial_spell=int(e.service_age[m]>=horizon)))
            for i,fs in frames.items():
                atomic_npz(target/f'trace_{chosen_scenes[i]}.npz',**{k:np.asarray([f[k] for f in fs]) for k in fs[0]},
                    bs_positions=np.asarray(cfg.env.bs_positions),width=cfg.env.width,height=cfg.env.height)
    array=np.concatenate(metrics);assert np.isfinite(array).all()
    atomic_npz(target/'step_metrics.npz',metrics=array,names=keys,scene_seeds=scenes)
    write_csv(target/'per_episode.csv',finalize_episodes(array,keys,scenes))
    write_csv(target/'per_step.csv',[dict(scene_seed=s,step=t+1,**dict(zip(keys,row)))
        for s,episode in zip(scenes,array) for t,row in enumerate(episode)])
    write_csv(target/'event_attribution.csv',events);write_csv(target/'service_spells.csv',spells)
    check_frozen(actor,meta)
    atomic_json(target/'provenance.json',dict(contract,ue_path_sha256=user_hash.hexdigest(),weights_unchanged=True,
        elapsed_seconds=time.perf_counter()-start,peak_rss_gib=peak/2**30,config=cfg.as_dict()))
    atomic_json(target/'complete.json',contract)
    print('complete',target,flush=True)


def calibrate(output,device='cuda:0',smoke=False):
    root=Path(output);path=root/'configs/locked_rules.json'
    if path.exists():print('rules already locked',locked(root),flush=True);return
    design=dict(calibration_scenes=CALIBRATION,confirmation_scenes=CONFIRMATION,
        calibration_action_seed=890001,grid=list(GRID),sample_every_steps=5,
        selection='maximum mean one-step task reward across both checkpoints; strictly positive beta/lambda; exact ties choose smaller beta then lambda',
        frozen=True,version=VERSION)
    atomic_json(root/'configs/preregistered_design.json',design)
    rows=[];qvalues=[];gaps=[]
    def accumulate(e,before,after,kind,scene,step):
        q=quality(e.pre_y);qvalues.extend(q[before].tolist())
        for i in range(len(before)):
            ids=np.flatnonzero(before[i]);rank=chosen(ids,q[i,ids],len(ids))
            if len(rank)>10:gaps.append(float(q[i,rank[9]]-q[i,rank[10]]))
        for beta in GRID:
            for lam in GRID:
                scores=rule_scores(e.pre_y,e.pre_c,e.cfg.capacity,beta,lam)
                result=resolve_topk(before,after,scores,e.cfg.capacity)
                r=outcome(*result,e.rates,np.where(e.pre_c.any(0),e.pre_c.argmax(0),-1),e.pre_age,
                    e.pre_last_owner,e.cfg.num_users,e.cfg.capacity)
                rows.append(dict(checkpoint=kind,scene_seed=scene,step=step,beta=beta,load_weight=lam,**r))
    for kind in ('best','last'):
        closed_case(kind,'R0',890001,root,device,'calibration',smoke,accumulate)
    if not rows:raise RuntimeError('Calibration results incomplete; choose a fresh smoke directory or restore saved calibration grid')
    write_csv(root/'configs/calibration_grid.csv',rows)
    means=[]
    for beta in GRID[1:]:
        for lam in GRID[1:]:
            r=[x for x in rows if x['beta']==beta and x['load_weight']==lam]
            means.append((float(np.mean([x['reward'] for x in r])),beta,lam))
    best=sorted(means,key=lambda x:(-x[0],x[1],x[2]))[0]
    quant=lambda x:{str(p):float(np.quantile(x,p)) for p in (.05,.25,.5,.75,.95)} if x else {}
    payload=dict(beta=best[1],load_weight=best[2],selection_reward=best[0],
        design=design,quality_quantiles=quant(qvalues),top10_11_quality_gap_quantiles=quant(gaps),
        grid_sha256=file_hash(root/'configs/calibration_grid.csv'),smoke=smoke)
    payload['contract_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    atomic_json(path,payload)
    print('LOCKED',payload,flush=True)


def unpack(x,shape=(18,150)):
    return np.unpackbits(x,count=int(np.prod(shape))).reshape(shape).astype(bool)


def candidate_matrix(obs,values,shape=(18,150)):
    result=np.zeros(shape,np.float64);bounds=np.r_[0,np.cumsum(obs.candidate_lengths)]
    for i in range(shape[0]):result[i,obs.candidate_ids[bounds[i]:bounds[i+1]]]=values[bounds[i]:bounds[i+1]]
    return result


def _extra_steps(case,begin,scenes):
    """One extra random transition per scene/stratum; exact inclusion probabilities."""
    eligible={s:{'11to15':[],'gt15':[]} for s in scenes}
    for path in sorted((case/'closed_loop').glob(f'b{begin}_t*.npz')):
        with np.load(path) as z:
            first=int(z['first_step'])
            for k in range(len(z['feasible_before'])):
                t=first+k
                if t in FIXED_STEPS:continue
                for i,s in enumerate(scenes):
                    n=unpack(z['feasible_before'][k,i]).sum(1)
                    if ((n>10)&(n<=15)).any():eligible[s]['11to15'].append(t)
                    if (n>15).any():eligible[s]['gt15'].append(t)
    selected={}; populations={}
    for s,by in eligible.items():
        for stratum,times in by.items():
            if times:
                populations[(s,stratum)]=len(times)
                rng=np.random.default_rng(np.random.SeedSequence([991001,s,0 if stratum=='11to15' else 1]))
                t=int(rng.choice(times));selected[(s,t,stratum)]=1/len(times)
    return selected,populations


def _shadow_row(e,before,after,scores,previous,ages,lastowner):
    parts=resolve_topk(before,after,scores,e.cfg.capacity)
    row=outcome(*parts,e.rates,previous,ages,lastowner,e.cfg.num_users,e.cfg.capacity)
    row.update(link_retained=int((parts[0]*e.pre_c).sum()),
        previous_links=int(e.pre_c.sum()),**competition_metrics(parts[2],scores,e.rates))
    return row,parts


def sample_snapshot(actor,obs,envs,prepared,contexts,output,identity,repeats=16,device='cuda:0'):
    """Sample joint actions, then project ONLY scores onto the recorded transition."""
    start=time.perf_counter()
    seed=int(np.random.SeedSequence([990001,0 if identity['checkpoint']=='best' else 1,
        identity['action_seed'],identity['begin'],identity['step']]).generate_state(1,np.uint64)[0]%(2**63-1))
    generator=torch.Generator(device=device).manual_seed(seed)
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None
    actions=[];latents=[]
    for _ in range(repeats):
        a,c=actor.sample(obs,generator=generator);actions.append(a);latents.append(c.scores[:,-1].copy())
    assert torch.equal(cpu_rng,torch.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    values=np.stack([a.scores for a in actions]);movement=np.stack([a.movement.reshape(-1,2) for a in actions])
    bounds=actions[0].offsets.copy();rows=[];edges=[];groups=[];shadow=[]
    for idx,(e,p,ctx) in enumerate(zip(envs,prepared,contexts)):
        base=dict(**ctx,**{k:identity[k] for k in ('checkpoint','action_seed','step')})
        low,hi=bounds[idx*18],bounds[(idx+1)*18]
        o=obs.take(np.arange(idx*18,(idx+1)*18))
        matrices=np.stack([candidate_matrix(o,x[low:hi]) for x in values])
        mean=matrices.mean(0);before,_,previous=p;after=e.diagnostic_after
        allparts=[]
        for rep,matrix in enumerate(matrices):
            r,parts=_shadow_row(e,before,after,matrix,previous,e.pre_age,e.pre_last_owner)
            shadow.append(dict(**base,score_mode='resample',repeat=rep,**r));allparts.append(parts)
        for mode,matrix in [('recorded',e.recorded_scores),('mean16',mean),('sinr',quality(e.pre_y))]:
            r,_=_shadow_row(e,before,after,matrix,previous,e.pre_age,e.pre_last_owner)
            shadow.append(dict(**base,score_mode=mode,repeat=-1,**r))
        for u in range(18):
            ids=np.flatnonzero(before[u]);n=len(ids)
            sets=[set(np.flatnonzero(a[1][u])) for a in allparts]
            jaccards=[len(a&b)/len(a|b) for j,a in enumerate(sets) for b in sets[j+1:] if a|b]
            probability=np.mean([a[1][u,ids] for a in allparts],axis=0) if n else np.empty(0)
            pairs=[];overlap=[]
            for j in ids:
                x=matrices[:,u,j]
                edges.append(dict(**base,uav=u,ue=int(j),score_mean=float(x.mean()),score_std=float(x.std()),
                    score_p05=float(np.quantile(x,.05)),score_p95=float(np.quantile(x,.95)),
                    selection_probability=float(np.mean([j in s for s in sets]))))
            if n>10:
                ordered=chosen(ids,mean[u,ids],n);j,k=ordered[9:11]
                boundary_flip=float((matrices[:,u,j]<matrices[:,u,k]).mean())
                overlap_interval=bool(np.quantile(matrices[:,u,j],.05)<=np.quantile(matrices[:,u,k],.95)
                    and np.quantile(matrices[:,u,k],.05)<=np.quantile(matrices[:,u,j],.95))
            else:boundary_flip=None;overlap_interval=None
            groups.append(dict(**base,uav=u,candidates=n,stratum='le10' if n<=10 else '11to15' if n<=15 else 'gt15',
                pairwise_topk_jaccard=float(np.mean(jaccards)) if jaccards else None,
                mean_score_std=float(matrices[:,u,ids].std(0).mean()) if n else None,
                ambiguous_selection_fraction=float(((probability>0)&(probability<1)).mean()) if n else None,
                boundary_rank_flip=boundary_flip,boundary_interval_overlap=overlap_interval,
                movement_vector_variance=float(movement[:,idx*18+u].var(0).sum())))
        r=[x for x in shadow if x['scene_seed']==ctx['scene_seed'] and x['score_mode']=='resample']
        rows.append(dict(**base,repeats=repeats,score_seed=seed,
            sampled_coverage_std=float(np.std([x['coverage'] for x in r])),
            sampled_throughput_std_mbps=float(np.std([x['throughput_bps'] for x in r])/1e6)))
    target=Path(output)/'score_samples'/f"b{identity['begin']}_t{identity['step']:03d}.npz"
    atomic_npz(target,scores=values,score_logits=np.stack(latents),movement=movement,
        candidate_ids=actions[0].ue_ids,offsets=bounds,scene_seeds=[x['scene_seed'] for x in contexts],
        fixed=obs.fixed,users=obs.users,observation_offsets=obs.offsets,ue_ids=obs.ue_ids,service_mask=obs.service_mask,
        recorded_movement=np.stack([e.recorded_movement for e in envs]),
        before=np.stack([p[0] for p in prepared]),after=np.stack([e.diagnostic_after for e in envs]),
        pre_sinr=np.stack([e.pre_y for e in envs]),post_rates=np.stack([e.rates for e in envs]),
        previous_c=np.stack([e.pre_c for e in envs]),recorded_scores=np.stack([e.recorded_scores for e in envs]),
        recorded_score_logits=np.stack([e.recorded_logits for e in envs]),
        score_seed=seed,elapsed_seconds=time.perf_counter()-start)
    return rows,groups,edges,shadow


@torch.no_grad()
def audit_s1(checkpoint,action_seed,output,device='cuda:0',smoke=False):
    cfg,actor,meta=source_actor(checkpoint,device);lock=locked(output)
    reference=resolve_output(REFERENCE)/'rollout'/checkpoint
    target=Path(output)/'s1'/checkpoint/str(action_seed)
    stamp=dict(meta,action_seed=action_seed,repeats=8 if smoke else 16,smoke=smoke,
        reference=str(reference),fixed_steps=list(FIXED_STEPS),rules_hash=lock['contract_sha256'])
    if (target/'complete.json').exists():
        assert json.loads((target/'complete.json').read_text())==stamp
        print('reuse',target,flush=True);return
    target.mkdir(parents=True,exist_ok=True);progress=Progress(25);started=time.perf_counter()
    rankings=[];snapshots=[];groups=[];edges=[];noise_shadow=[];shadows=[];events=[];spells=[];loads=[]
    audits=[];block_manifest=[];peak=0
    for protocol in ('T1','T2'):
        case=reference/protocol/str(action_seed)
        old=json.loads((case/'provenance.json').read_text())
        assert old['checkpoint_sha256']==meta['checkpoint_sha256']
        replay_metrics=[];metric_keys=None;user_hash=hashlib.sha256();sample_seconds=0.
        for begin in (0,16):
            if smoke and begin:continue
            initial=torch.load(reference/'T1'/str(action_seed)/f'initial_snapshots_{begin}.pt',map_location='cpu',weights_only=False)
            scenes=initial['scenes'];envs=[ScoreRuleEnvironment(cfg.env,'T1' if protocol=='T1' else 'R0') for _ in scenes]
            for e,snapshot in zip(envs,initial['snapshots']):e.restore(snapshot)
            user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
            extra,populations=_extra_steps(case,begin,scenes) if protocol=='T1' and not smoke else ({},{})
            batch=[]
            for start in range(0,400,50):
                if smoke and start:break
                block_path=case/'closed_loop'/f'b{begin}_t{start:03d}.npz'
                block_manifest.append(dict(path=str(block_path),sha256=file_hash(block_path)))
                with np.load(block_path) as z:closed={k:z[k] for k in z.files}
                raw=None
                if protocol=='T1':
                    raw_path=case/'raw'/f'b{begin}_t{start:03d}.npz'
                    block_manifest.append(dict(path=str(raw_path),sha256=file_hash(raw_path)))
                    with np.load(raw_path) as z:raw={k:z[k] for k in z.files}
                for j in range(50):
                    t=start+j+1
                    if smoke and t>5:break
                    selected=[];contexts=[];preobs=[];items=[];obslist=[]
                    for i,(e,s) in enumerate(zip(envs,scenes)):
                        o=e.obs.copy();obslist.append(o)
                        e.recorded_scores=closed['scores'][j,i].copy();e.recorded_movement=closed['movement'][j,i].copy()
                        learned=candidate_matrix(o,np.zeros(len(o.candidate_ids)))
                        a=ServiceAction(e.recorded_movement,
                            np.concatenate([e.recorded_scores[u,o.candidate_ids[np.r_[0,np.cumsum(o.candidate_lengths)][u]:np.r_[0,np.cumsum(o.candidate_lengths)][u+1]]]
                                for u in range(18)]),o.candidate_ids.copy(),np.r_[0,np.cumsum(o.candidate_lengths)])
                        items.append(a)
                        if protocol=='T1':
                            e.recorded_logits=np.full((18,150),np.nan)
                            offsets=raw['chain_candidate_offsets'][j]
                            chain_start=raw['chain_score_offsets'][j]
                            id_start=raw['candidate_id_offsets'][j]
                            np.testing.assert_array_equal(o.candidate_ids,
                                raw['candidate_ids'][id_start+offsets[i*18]:id_start+offsets[(i+1)*18]])
                            for u in range(18):
                                lo,hi=offsets[i*18+u:i*18+u+2]
                                ids=o.candidate_ids[a.offsets[u]:a.offsets[u+1]]
                                vals=raw['chain_scores'][chain_start+lo:chain_start+hi,-1]
                                assert len(vals)==len(ids)
                                e.recorded_logits[u,ids]=vals
                                # Float32 sigmoid execution should match stored logits; allow rounding.
                                np.testing.assert_allclose(a.scores[a.offsets[u]:a.offsets[u+1]],
                                    np.exp(-np.logaddexp(0.,-vals)),atol=1e-7,rtol=1e-6)
                        is_fixed=t in FIXED_STEPS or (smoke and t==5)
                        picked=[(key[2],prob) for key,prob in extra.items() if key[:2]==(s,t)]
                        if protocol=='T1' and (is_fixed or picked):
                            selected.append(i);preobs.append(o)
                            counts=o.candidate_lengths
                            pa=(1/populations[(s,'11to15')]) if ((counts>10)&(counts<=15)).any() and (s,'11to15') in populations else 0.
                            pb=(1/populations[(s,'gt15')]) if (counts>15).any() and (s,'gt15') in populations else 0.
                            inclusion=1. if is_fixed else 1-(1-pa)*(1-pb)
                            contexts.append(dict(scene_seed=s,pool='fixed' if is_fixed else 'crowded_extra',
                                extra_strata='|'.join(x[0] for x in picked),
                                inclusion_probability=inclusion,design_weight=1/inclusion))
                    prepared=[e.prepare(a) for e,a in zip(envs,items)]
                    links=physical_links(cfg.env,np.stack([e.uav_xy for e in envs]),np.stack([e.user_xy for e in envs]))
                    results=[e.finish(p,tuple(x[i] for x in links)) for i,(e,p) in enumerate(zip(envs,prepared))]
                    user_hash.update(np.stack([e.user_xy for e in envs]).tobytes())
                    infos=[numeric_info(e,r) for e,r in zip(envs,results)]
                    if metric_keys is None:metric_keys=sorted(k for k,v in infos[0].items() if np.isscalar(v))
                    batch.append([[r[k] for k in metric_keys] for r in infos])
                    for i,(e,s,p) in enumerate(zip(envs,scenes,prepared)):
                        np.testing.assert_array_equal(e.C,unpack(closed['association'][j,i]))
                        np.testing.assert_array_equal(e.uav_xy,closed['uav_xy'][j,i])
                        np.testing.assert_array_equal(e.user_xy,closed['user_xy'][j,i])
                        validate_reward(e)
                        append_spells(spells,e,p[2],e.pre_age,s,t)
                        # Track protocol explicitly, so T1 and T2 spells never pool silently.
                        for r in spells[::-1]:
                            if 'protocol' in r:break
                            r['protocol']=protocol
                        if t>=301:
                            for r in interruption_rows(e.pre_c,e.C,p[0],e.diagnostic_after,e.proposals,e.B,e.pre_y,e.Y,e.pre_age,10):
                                events.append(dict(protocol=protocol,scene_seed=s,step=t,**r))
                        loads.append(dict(protocol=protocol,scene_seed=s,step=t,
                            **{f'load_{n}':int((e.pre_c.sum(1)==n).sum()) for n in range(11)},
                            crowded_uavs=int((p[0].sum(1)>10).sum()),candidate_edges=int(p[0].sum()),
                            multi_candidate_users=int((p[0].sum(0)>1).sum())))
                        if (t in FIXED_STEPS or (smoke and t==5)) and protocol=='T1':
                            for r in ranking_rows(p[0],e.recorded_scores,e.pre_y,e.rates,e.diagnostic_after,10):
                                rankings.append(dict(scene_seed=s,step=t,pool='fixed',inclusion_probability=1.,design_weight=1.,**r))
                        elif protocol=='T1' and i in selected:
                            for r in ranking_rows(p[0],e.recorded_scores,e.pre_y,e.rates,e.diagnostic_after,10):
                                stratum=r['stratum'];prob=extra.get((s,t,stratum))
                                if prob is not None:
                                    rankings.append(dict(scene_seed=s,step=t,pool='crowded_extra',inclusion_probability=prob,design_weight=1/prob,**r))
                        if t in FIXED_STEPS or (smoke and t==5):
                            for rule,beta,lam in [('R0',0.,0.),('R1',lock['beta'],0.),('R2',0.,lock['load_weight']),('R3',lock['beta'],lock['load_weight'])]:
                                scores=rule_scores(e.pre_y,e.pre_c,10,beta,lam)
                                row,_=_shadow_row(e,p[0],e.diagnostic_after,scores,p[2],e.pre_age,e.pre_last_owner)
                                shadows.append(dict(reference_pool=protocol,scene_seed=s,step=t,rule=rule,**row))
                            row,_=_shadow_row(e,p[0],e.diagnostic_after,e.recorded_scores,p[2],e.pre_age,e.pre_last_owner)
                            shadows.append(dict(reference_pool=protocol,scene_seed=s,step=t,rule='recorded',**row))
                    if selected:
                        st=time.perf_counter()
                        r,g,ed,sh=sample_snapshot(actor,stack(preobs),[envs[i] for i in selected],
                            [prepared[i] for i in selected],contexts,target,
                            dict(checkpoint=checkpoint,action_seed=action_seed,begin=begin,step=t),8 if smoke else 16,device)
                        sample_seconds+=time.perf_counter()-st
                        snapshots.extend(r);groups.extend(g);edges.extend(ed);noise_shadow.extend(sh)
                    peak=max(peak,psutil.Process().memory_info().rss)
                    if peak>12*2**30:raise MemoryError('S1 exceeds 12 GiB RSS')
                    progress(f'S1/{checkpoint}/{action_seed}/{protocol} batch={begin} t={t}/400 snapshots={len(snapshots)}')
            replay_metrics.append(np.asarray(batch,float).transpose(1,0,2))
            for e,s in zip(envs,scenes):
                for m in np.flatnonzero((e.owner()>=0)&(e.service_age>0)):
                    spells.append(dict(scene_seed=s,ue=int(m),end_step=e.t,duration=int(e.service_age[m]),
                        event=0,initial_spell=int(e.service_age[m]>=e.t),protocol=protocol))
        array=np.concatenate(replay_metrics)
        regression={}
        if not smoke:
            with np.load(case/'step_metrics.npz') as z:
                for key in ('coverage','throughput_bps','reward','service_interruptions','direct_switches'):
                    err=float(np.max(np.abs(array[:,:,metric_keys.index(key)]-z['metrics'][:,:,z['names'].tolist().index(key)])))
                    regression[key]=err
                    if err>1e-6:raise AssertionError(f'Replay differs: {protocol} {key} {err}')
            assert user_hash.hexdigest()==old['ue_path_sha256']
        atomic_npz(target/f'{protocol}_step_metrics.npz',metrics=array,names=metric_keys,scene_seeds=SCENES[:len(array)])
        write_csv(target/f'{protocol}_per_episode.csv',finalize_episodes(array,metric_keys,SCENES[:len(array)]))
        audits.append(dict(protocol=protocol,max_absolute_errors=regression,ue_path_sha256=user_hash.hexdigest(),
            sampled_seconds=sample_seconds,original_source_sha256=file_hash(case/'provenance.json')))
    write_csv(target/'snapshots.csv',snapshots);write_csv(target/'ranking_metrics.csv',rankings)
    write_csv(target/'noise_group_metrics.csv',groups);write_csv(target/'score_edge_summary.csv',edges)
    write_csv(target/'noise_shadow.csv',noise_shadow);write_csv(target/'shadow_paired.csv',shadows)
    write_csv(target/'event_attribution.csv',events);write_csv(target/'service_spells.csv',spells)
    write_csv(target/'load_structure.csv',loads)
    atomic_json(target/'source_blocks.json',block_manifest)
    check_frozen(actor,meta)
    atomic_json(target/'provenance.json',dict(stamp,weights_unchanged=True,peak_rss_gib=peak/2**30,
        elapsed_seconds=time.perf_counter()-started,replay_regression=audits,
        score_format='Ragged per-snapshot NPZ: all final motion, score logits and sigmoid scores; no padding'))
    atomic_json(target/'complete.json',stamp)
    print('complete',target,flush=True)
