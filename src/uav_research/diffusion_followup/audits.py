"""Pre-registered scale and coupling audits, with independent non-validation scenes."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from ..runtime import isolated_rng,seed_all,atomic_json,atomic_save
from ..diffusion_dppo.policy import make_service_actor
from ..diffusion_dppo.environment import ServiceEnvironment
from ..diffusion_dppo.observations import stack
from ..diffusion_dppo.config import contract
from ..diffusion_dppo.storage import configure,load
from ..topk_diagnostic import atomic_npz,file_hash
from ..system_ablation.report import write_csv
from .runtime import ROOT,resolve,actor_digest


def observation_pools(cfg,independent=128):
    training=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence([cfg.run.seed,101]).spawn(cfg.run.num_envs)]
    seeds=training+list(range(932000,932000+independent))
    observations=[]
    for seed in seeds:
        env=ServiceEnvironment(cfg.env)
        obs,_,_=env.reset(seed);observations.append(obs)
    return stack(observations),seeds,np.repeat([0]*len(training)+[1]*independent,cfg.env.num_uavs)


@torch.no_grad()
def sample_scale(actor,obs,pool_ids,repeats=64,batch=256,seed=993001):
    device=next(actor.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    mnorm,anorm,slatent,score_pools,m_pools=[],[],[],[],[]
    profiles=[];max_error=0.;max_excess=-float('inf');finite=True
    for repeat in range(repeats):
        if repeat%8==0:print(f'scale audit repeat={repeat+1}/{repeats} local_observations={obs.groups}',flush=True)
        for lo in range(0,obs.groups,batch):
            ids=np.arange(lo,min(obs.groups,lo+batch));chosen=obs.take(ids)
            action,chain=actor.sample(chosen,generator=generator)
            assert np.isfinite(action.movement).all() and np.isfinite(action.scores).all()
            assert np.linalg.norm(action.movement.reshape(-1,2),axis=-1).max()<=1.000001
            assert np.all((action.scores>=0)&(action.scores<=1))
            np.testing.assert_array_equal(action.ue_ids,chosen.candidate_ids)
            # Same saved chain, no resampling and no post-hoc latent clipping.
            recomputed=actor.evaluate_chain(chosen,chain).cpu().numpy()
            diff=np.abs(recomputed-chain.logp)
            max_error=max(max_error,float(diff.max(initial=0)))
            max_excess=max(max_excess,float((diff-(.002+.0002*np.abs(chain.logp))).max(initial=-np.inf)))
            finite=finite and np.isfinite(recomputed).all() and np.isfinite(chain.logp).all()
            if chain.movement_logp is not None:
                np.testing.assert_allclose(chain.logp,chain.movement_logp+chain.score_logp,atol=2e-3,rtol=2e-4)
            mnorm.extend(np.linalg.norm(chain.movement[:,-1],axis=-1).tolist())
            anorm.extend(np.linalg.norm(action.movement.reshape(-1,2),axis=-1).tolist())
            m_pools.extend(pool_ids[ids].tolist())
            slatent.extend(chain.scores[:,-1].tolist())
            score_pools.extend(np.repeat(pool_ids[ids],chosen.candidate_lengths).tolist())
            # One profile per sampled group/transition, averaged later.
            profile=dict(repeat=repeat,groups=len(ids),movement_chain_mean=chain.movement.mean((0,2)).tolist(),
                movement_chain_second_moment=np.square(chain.movement).mean((0,2)).tolist(),
                logp_mean=chain.logp.mean(0).tolist(),
                score_chain_mean=chain.scores.mean(0).tolist() if len(chain.scores) else None,
                score_chain_second_moment=np.square(chain.scores).mean(0).tolist() if len(chain.scores) else None,
                coordinates=len(chain.scores),recomputed_kl_mean=float((np.expm1(recomputed-chain.logp)-(recomputed-chain.logp)).mean()),
                recomputed_clip_fraction=float((np.abs(np.expm1(recomputed-chain.logp))>.1).mean()))
            profiles.append(profile)
    arrays=dict(movement_norm=np.asarray(mnorm),action_norm=np.asarray(anorm),
        score_latent=np.asarray(slatent),movement_pool=np.asarray(m_pools),score_pool=np.asarray(score_pools))
    summaries={}
    for pool in (-1,0,1):
        m=arrays['movement_norm'] if pool==-1 else arrays['movement_norm'][arrays['movement_pool']==pool]
        a=arrays['action_norm'] if pool==-1 else arrays['action_norm'][arrays['movement_pool']==pool]
        s=arrays['score_latent'] if pool==-1 else arrays['score_latent'][arrays['score_pool']==pool]
        q=np.exp(-np.logaddexp(0.,-s))
        summaries[str(pool)]=dict(movement_latent_norm_mean=float(m.mean()),
            movement_latent_norm_p50=float(np.quantile(m,.5)),movement_latent_norm_p90=float(np.quantile(m,.9)),
            movement_latent_norm_p99=float(np.quantile(m,.99)),movement_norm_mean=float(a.mean()),
            movement_norm_p95=float(np.quantile(a,.95)),score_latent_std=float(s.std()) if len(s) else None,
            score_saturation_01=float(((q<.01)|(q>.99)).mean()) if len(s) else None,
            score_saturation_05=float(((q<.05)|(q>.95)).mean()) if len(s) else None)
    return dict(pools=summaries,finite=bool(finite),logp_max_abs_error=max_error,
                logp_max_tolerance_excess=max_excess,profiles=profiles),arrays


def initialization_audit(output,device='cpu',repeats=64,independent=128,batch=256):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg,_,_=resolve('A1b',device);configure(cfg)
    manifest=json.loads((ROOT/'configs/diffusion_followup_1m/manifest.json').read_text())
    thresholds=manifest['init_audit']['thresholds']
    with isolated_rng():
        obs,seeds,pools=observation_pools(cfg,independent)
        atomic_save(output/'observation_pool.pt',dict(observation=obs,seeds=seeds,pool_ids=pools))
        results={}
        for name in ('A1','G0','A1b'):
            print(f'initialization audit model={name} repeats={repeats} independent_scenes={independent}',flush=True)
            if name=='A1b':chosen=cfg
            else:
                from ..config import load_config
                chosen=load_config(ROOT/f'configs/system_ablation_1m/{name}.yaml');chosen.run.device=device
            seed_all(chosen.run.seed)
            actor=make_service_actor(chosen).to(device).eval()
            summary,arrays=sample_scale(actor,obs,pools,repeats,batch)
            summary['actor_sha256']=actor_digest(actor)
            results[name]=summary
            atomic_npz(output/f'{name}_samples.npz',**arrays)
            atomic_json(output/f'{name}_summary.json',summary)
        stable=results['A1b']
        violations=[]
        for name,s in stable['pools'].items():
            for metric,key in [('movement_latent_norm_mean','latent_norm_mean_max'),
                ('movement_latent_norm_p99','latent_norm_p99_max'),('movement_norm_mean','movement_norm_mean_max'),
                ('movement_norm_p95','movement_norm_p95_max'),('score_saturation_01','score_saturation_01_max')]:
                if s[metric] is None or s[metric]>thresholds[key]:violations.append(f'pool{name}:{metric}')
            if (s['movement_latent_norm_mean']<thresholds['latent_norm_mean_min'] or
                    s['movement_norm_mean']<thresholds['movement_norm_mean_min'] or
                    s['score_latent_std'] is None or s['score_latent_std']<thresholds['score_latent_std_min']):
                violations.append(f'pool{name}:collapsed exploration')
        if not stable['finite'] or stable['logp_max_tolerance_excess']>0:
            violations.append('probability consistency')
        full=repeats>=64 and independent>=128
        gate=dict(passed=not violations and full,full_audit=full,violations=violations,
            repeats=repeats,independent_scenes=independent,training_scenes=seeds[:cfg.run.num_envs],
            independent_scene_seeds=seeds[cfg.run.num_envs:],thresholds=thresholds,
            actor_sha256=stable['actor_sha256'],contract=contract(cfg),
            calibration='No validation/test reward used; thresholds frozen before sampling')
        atomic_json(output/'gate.json',gate)
    return gate


@torch.no_grad()
def coupling_audit(output,device='cpu'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg,_,_=resolve('F2c',device);configure(cfg)
    from ..runtime import resolve_output
    path=resolve_output('runs/system_ablation_1m/A3_seed0/step0.pt')
    source=load(path,kinds={'dppo_model'})
    from ..diffusion_dppo.config import from_dict
    original_cfg=from_dict(source['config']);original_cfg.run.device=device
    with isolated_rng():
        seed_all(993101)
        original=make_service_actor(original_cfg).to(device)
        gated=make_service_actor(cfg).to(device)
        original.load_state_dict(source['learner']['actor']);gated.load_state_dict(source['learner']['actor'])
        obs,_,_=observation_pools(cfg,independent=4);obs=obs.take(np.arange(min(128,obs.groups)))
        results=[]
        for step in range(cfg.diffusion.steps):
            m=torch.randn((obs.groups,2),device=device)
            s=torch.randn(int(obs.candidate_lengths.sum()),device=device)
            perturbed=s+torch.randn_like(s)
            for name,actor in [('A3_same_BC_weights',original),('F2c_gate0_same_weights_diagnostic_only',gated)]:
                enc=actor.encode(obs.tensor(device))
                a,b=actor.predict_noise(enc,m,s,step);c,d=actor.predict_noise(enc,m,perturbed,step)
                # Ensure observation context is still a live input.
                changed=copy.copy(enc);changed.context=enc.context+.1
                e,_=actor.predict_noise(changed,m,s,step)
                results.append(dict(model=name,step=step+1,motion_score_perturb_max=float((a-c).abs().max()),
                    score_prediction_perturb_mean=float((b-d).abs().mean()),
                    motion_context_perturb_mean=float((a-e).abs().mean())))
        gated_rows=[r for r in results if r['model'].startswith('F2c')]
        assert all(r['motion_score_perturb_max']==0 for r in gated_rows)
        assert any(r['motion_context_perturb_mean']>0 for r in gated_rows)
        write_csv(output/'sensitivity.csv',results)
        result=dict(passed=True,source_sha256=file_hash(path),same_weights_for_audit_only=True,
            formal_F2c_requires_new_BC=True,parameter_count_A3=sum(p.numel() for p in original.parameters()),
            parameter_count_F2c=sum(p.numel() for p in gated.parameters()))
        atomic_json(output/'summary.json',result)
        return result
