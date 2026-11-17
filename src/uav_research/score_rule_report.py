"""Saved-data-only S1/S2 summaries, paired scene inference, and figure rebuild."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .score_rule_experiment import (read_csv,locked,SCENES,CONFIRMATION,ACTION_SEEDS,
    REFERENCE,source_actor,check_frozen)
from .score_rule_diagnostic import CAUSES,survival_rows
from .association_timing_experiment import bootstrap
from .system_ablation.report import write_csv
from .runtime import atomic_json,resolve_output
from .topk_diagnostic import file_hash

COLORS={'T1':'#333333','R0':'#0072B2','R1':'#009E73','R2':'#E69F00','R3':'#D55E00',
        'recorded':'#333333','mean16':'#009E73','sinr':'#0072B2','resample':'#E69F00'}
STYLES={'T1':':','R0':'-','R1':'--','R2':'-.','R3':':'}
PRIMARY=('last100_coverage','last50_coverage','final_coverage','last100_throughput_mbps',
    'last50_throughput_mbps','final_throughput_mbps','last100_coverage_std_pp',
    'last100_coverage_p05','last100_drop_events_10pp','last100_max_drop_pp','episode_return',
    'last100_service_interruptions_window_sum','last100_direct_switches_window_sum',
    'last100_service_reconnections_window_sum','last100_link_retention',
    'last100_connected_fraction','time_to_60pct_10steps','not_reached_60pct_10steps',
    'last100_idle_slots','last100_full_uav_fraction','last100_over_capacity_fraction',
    'last100_multi_candidate_fraction','last100_proposal_acceptance',
    'last100_served_ue_rate_mean_mbps','last100_served_ue_rate_p05_mbps')


def numeric(row):
    out={}
    for k,v in row.items():
        if v is None or v=='':out[k]=None;continue
        try:out[k]=float(v)
        except (ValueError,TypeError):out[k]=v
    return out


def finite(values):
    return [float(x) for x in values if x is not None and np.isfinite(x)]


def mean(values):
    values=finite(values)
    return float(np.mean(values)) if values else None


def _scene_average(rows,metric):
    out={}
    for s in sorted({int(r['scene_seed']) for r in rows}):
        out[s]=mean([r.get(metric) for r in rows if int(r['scene_seed'])==s])
    return out


def pair_value(left,right,metric):
    a,b=_scene_average(left,metric),_scene_average(right,metric)
    scenes=sorted(set(a)&set(b))
    values=[a[s]-b[s] for s in scenes if a[s] is not None and b[s] is not None]
    if not values:return None
    return dict(zip(('difference','low','high'),bootstrap(values)),scenes=len(values))


def s1_tables(root):
    ranks=[];noise=[];edges=[];snapshots=[];shadow=[];rules=[];events=[];spells=[];loads=[]
    for path in sorted((root/'s1').glob('*/*/complete.json')):
        stamp=json.loads(path.read_text());case=path.parent
        prefix=dict(checkpoint=stamp['checkpoint'],action_seed=stamp['action_seed'])
        for filename,dest in [('ranking_metrics',ranks),('noise_group_metrics',noise),
            ('score_edge_summary',edges),('snapshots',snapshots),('noise_shadow',shadow),
            ('shadow_paired',rules),('event_attribution',events),('service_spells',spells),('load_structure',loads)]:
            for r in read_csv(case/(filename+'.csv')):dest.append(dict(prefix,**numeric(r)))
    for name,rows in [('snapshots',snapshots),('ranking_metrics',ranks),('noise_group_metrics',noise),
        ('score_edge_summary',edges),('noise_shadow',shadow),('event_attribution',events),('service_spells',spells),
        ('load_structure',loads)]:write_csv(root/'s1'/(name+'.csv'),rows)
    write_csv(root/'s2/shadow_paired.csv',rules)
    rank_summary=[];noise_summary=[];shadow_summary=[];effects=[];event_summary=[];survival=[]
    for checkpoint in ('best','last'):
        for pool in ('fixed','crowded_extra'):
            for stratum in ('all','crowded','le10','11to15','gt15'):
                select=lambda r:r['checkpoint']==checkpoint and r['pool']==pool and (stratum=='all' or
                    (r['candidates']>10 if stratum=='crowded' else r['stratum']==stratum))
                a=[r for r in ranks if select(r)]
                if a:
                    row=dict(checkpoint=checkpoint,pool=pool,stratum=stratum,uav_states=len(a),
                        defined_correlations=len(finite([r.get('spearman') for r in a])),
                        **{k:mean([r.get(k) for r in a]) for k in ('spearman','rate_spearman','topk_jaccard','topk_overlap',
                         'high_quality_missed','learned_post_valid','quality_post_valid','usable_alternative_users')})
                    # Extra data have unequal sampling probabilities. Their
                    # descriptive estimate uses inverse inclusion weights and
                    # remains separate from fixed-time primary estimates.
                    if pool=='crowded_extra':
                        for k in ('spearman','topk_jaccard','high_quality_missed'):
                            rr=[r for r in a if r.get(k) is not None]
                            row[k+'_ipw']=float(np.average([r[k] for r in rr],weights=[r['design_weight'] for r in rr])) if rr else None
                    rank_summary.append(row)
                b=[r for r in noise if select(r)]
                if b:
                    noise_summary.append(dict(checkpoint=checkpoint,pool=pool,stratum=stratum,uav_states=len(b),
                        **{k:mean([r.get(k) for r in b]) for k in ('pairwise_topk_jaccard','mean_score_std',
                        'ambiguous_selection_fraction','boundary_rank_flip','movement_vector_variance')},
                        boundary_interval_overlap=mean([float(r['boundary_interval_overlap']=='True') if isinstance(r.get('boundary_interval_overlap'),str)
                            else r.get('boundary_interval_overlap') for r in b])))
        fixed=[r for r in shadow if r['checkpoint']==checkpoint and r['pool']=='fixed']
        # Mean 16 resamples within each snapshot before scene averaging.
        aggregated=[]
        for key in sorted({(r['scene_seed'],r['action_seed'],r['step'],r['score_mode']) for r in fixed}):
            aa=[r for r in fixed if (r['scene_seed'],r['action_seed'],r['step'],r['score_mode'])==key]
            aggregated.append(dict(scene_seed=key[0],action_seed=key[1],step=key[2],score_mode=key[3],
                **{k:mean([r[k] for r in aa]) for k in ('coverage','throughput_bps','reward','service_interruptions',
                    'competing_users','competing_rank_count','competing_rank_sum','competing_best_rate_agreements')}))
        for mode in ('recorded','resample','mean16','sinr'):
            aa=[r for r in aggregated if r['score_mode']==mode]
            row=dict(checkpoint=checkpoint,score_mode=mode,snapshots=len(aa))
            for k in ('coverage','throughput_bps','reward','service_interruptions'):
                row[k]=mean(list(_scene_average(aa,k).values()))
            den=sum(r['competing_users'] for r in aa)
            row['competing_user_cases']=den
            row['best_rate_choice_agreement']=sum(r['competing_best_rate_agreements'] for r in aa)/den if den else None
            den2=sum(r['competing_rank_count'] for r in aa)
            row['cross_uav_spearman']=sum(r['competing_rank_sum'] for r in aa)/den2 if den2 else None
            shadow_summary.append(row)
            if mode!='recorded':
                for metric in ('coverage','throughput_bps','reward','service_interruptions'):
                    diff=pair_value(aa,[r for r in aggregated if r['score_mode']=='recorded'],metric)
                    effects.append(dict(checkpoint=checkpoint,comparison=mode+' - recorded',metric=metric,**diff))
        for protocol in ('T1','T2'):
            ev=[r for r in events if r['checkpoint']==checkpoint and r['protocol']==protocol]
            for cause in CAUSES:
                count=sum(r['cause']==cause for r in ev)
                event_summary.append(dict(checkpoint=checkpoint,protocol=protocol,cause=cause,events=count,
                    fraction=count/len(ev) if ev else 0,per_episode_last100=count/(32*3)))
            sp=[r for r in spells if r['checkpoint']==checkpoint and r['protocol']==protocol]
            km=survival_rows([r['duration'] for r in sp],[r['event'] for r in sp])
            survival.extend(dict(checkpoint=checkpoint,protocol=protocol,**r) for r in km)
    write_csv(root/'s1/ranking_summary.csv',rank_summary);write_csv(root/'s1/noise_summary.csv',noise_summary)
    write_csv(root/'s1/score_effect_summary.csv',shadow_summary);write_csv(root/'s1/score_effect_paired.csv',effects)
    write_csv(root/'s1/event_summary.csv',event_summary);write_csv(root/'s1/survival.csv',survival)
    shadow_effects=[]
    for checkpoint in ('best','last'):
        for pool in ('T1','T2'):
            rr=[r for r in rules if r['checkpoint']==checkpoint and r['reference_pool']==pool]
            for left,right in [('R0','recorded'),('R1','R0'),('R2','R0'),('R3','R0')]:
                for metric in ('coverage','throughput_bps','reward','service_interruptions','direct_switches'):
                    diff=pair_value([r for r in rr if r['rule']==left],[r for r in rr if r['rule']==right],metric)
                    shadow_effects.append(dict(checkpoint=checkpoint,reference_pool=pool,comparison=left+' - '+right,metric=metric,**diff))
    write_csv(root/'s2/shadow_effects.csv',shadow_effects)
    return rank_summary,noise_summary,shadow_summary,effects,event_summary,survival


def s2_tables(root):
    episodes=[];sources=[];series={}
    # Baselines are independently replayed and checked to match old T1/T2.
    for stamp_path in sorted((root/'s1').glob('*/*/complete.json')):
        stamp=json.loads(stamp_path.read_text());case=stamp_path.parent
        for protocol,rule in [('T1','T1'),('T2','R0')]:
            for r in read_csv(case/f'{protocol}_per_episode.csv'):
                episodes.append(dict(checkpoint=stamp['checkpoint'],action_seed=stamp['action_seed'],rule=rule,
                    split='development',**numeric(r)))
            with np.load(case/f'{protocol}_step_metrics.npz') as z:
                series[('development',stamp['checkpoint'],rule,stamp['action_seed'])]=(z['metrics'].copy(),z['names'].tolist())
    for path in sorted((root/'s2/rollout').glob('*/*/*/*/complete.json')):
        stamp=json.loads(path.read_text())
        if stamp['split']=='calibration':continue
        case=path.parent
        # Extra R0 development regression should not duplicate the replay baseline.
        if stamp['split']=='development' and stamp['rule']=='R0':continue
        sources.append(json.loads((case/'provenance.json').read_text()))
        for r in read_csv(case/'per_episode.csv'):
            episodes.append(dict(checkpoint=stamp['checkpoint'],rule=stamp['rule'],split=stamp['split'],
                action_seed=stamp['action_seed'],**numeric(r)))
        with np.load(case/'step_metrics.npz') as z:
            series[(stamp['split'],stamp['checkpoint'],stamp['rule'],stamp['action_seed'])]=(z['metrics'].copy(),z['names'].tolist())
    write_csv(root/'s2/per_episode.csv',episodes)
    summary=[];paired=[];scene_rows=[];groups={}
    for key in sorted({(r['split'],r['checkpoint'],r['rule']) for r in episodes}):
        rr=[r for r in episodes if (r['split'],r['checkpoint'],r['rule'])==key]
        groups[key]=rr;row=dict(split=key[0],checkpoint=key[1],rule=key[2],scenes=len({r['scene_seed'] for r in rr}),
                             action_repetitions=len({r['action_seed'] for r in rr}))
        metrics=[k for k in rr[0] if k not in ('scene_seed','checkpoint','rule','split','action_seed')]
        per_scene={metric:_scene_average(rr,metric) for metric in metrics}
        for metric in metrics:
            values=finite(list(per_scene[metric].values()))
            if values:
                avg,low,high=bootstrap(values);row.update({metric:avg,metric+'_ci_low':low,metric+'_ci_high':high})
            else:row[metric]=None
        for s in sorted({r['scene_seed'] for r in rr}):
            scene_rows.append(dict(split=key[0],checkpoint=key[1],rule=key[2],scene_seed=s,
                **{k:v[s] for k,v in per_scene.items()}))
        summary.append(row)
    for split in ('development','confirmation'):
        for ck in ('best','last'):
            for left,right in [('R0','T1'),('R1','R0'),('R2','R0'),('R3','R0'),('R3','R1'),('R3','R2')]:
                a,b=groups.get((split,ck,left)),groups.get((split,ck,right))
                if a is None or b is None:continue
                for metric in PRIMARY:
                    val=pair_value(a,b,metric)
                    if val:paired.append(dict(split=split,checkpoint=ck,comparison=left+' - '+right,metric=metric,**val))
            sets=[groups.get((split,ck,r)) for r in ('R3','R2','R1','R0')]
            if any(x is None for x in sets):continue
            for metric in PRIMARY:
                aa=[_scene_average(x,metric) for x in sets];ss=sorted(set.intersection(*[set(x) for x in aa]))
                x=[aa[0][s]-aa[1][s]-aa[2][s]+aa[3][s] for s in ss if all(z[s] is not None for z in aa)]
                if x:paired.append(dict(split=split,checkpoint=ck,comparison='interaction',metric=metric,
                    **dict(zip(('difference','low','high'),bootstrap(x))),scenes=len(x)))
    write_csv(root/'s2/comparison.csv',summary);write_csv(root/'s2/paired_differences.csv',paired)
    write_csv(root/'s2/per_scene.csv',scene_rows);atomic_json(root/'audit/closed_sources.json',sources)
    # Full step CSV lives in individual cases. Build one streaming table without
    # holding millions of duplicate dictionaries in RAM.
    destination=root/'s2/per_step.csv';schema=None
    with destination.open('w',newline='') as output:
        writer=None
        for (split,ck,rule,seed),(array,names) in sorted(series.items()):
            if schema is None:
                schema=['split','checkpoint','rule','action_seed','scene_seed','step']+names
                writer=csv.DictWriter(output,fieldnames=schema);writer.writeheader()
            if names!=schema[6:]:raise AssertionError('Step schemas differ')
            ss=SCENES if split=='development' else CONFIRMATION
            for s,episode in zip(ss,array):
                for t,x in enumerate(episode):
                    writer.writerow(dict(split=split,checkpoint=ck,rule=rule,action_seed=seed,scene_seed=s,
                        step=t+1,**dict(zip(names,x))))
    return summary,paired,series


def export_parquet(root):
    """Optional local-only columnar export; original NPZ is sufficient for rebuild."""
    try:import pyarrow as pa;import pyarrow.parquet as pq
    except ImportError:return False
    target=root/'s1/score_samples.parquet';writer=None
    try:
        for path in sorted((root/'s1').glob('*/*/score_samples/*.npz')):
            checkpoint=path.parents[2].name;seed=int(path.parents[1].name)
            with np.load(path) as z:
                score=z['scores'];bounds=z['offsets'];ids=z['candidate_ids'];scenes=z['scene_seeds'];
                owners=np.repeat(np.arange(len(bounds)-1),np.diff(bounds))
                if not len(ids):continue
                step=int(path.stem.split('_t')[-1]);k,n=score.shape
                rows=dict(checkpoint=np.repeat(checkpoint,k*n),action_seed=np.full(k*n,seed,np.int64),
                    step=np.full(k*n,step,np.int32),scene_seed=np.tile(scenes[owners//18],k).astype(np.int32),
                    uav=np.tile(owners%18,k).astype(np.int16),ue=np.tile(ids,k).astype(np.int16),
                    repeat=np.repeat(np.arange(k,dtype=np.int16),n),score=score.reshape(-1),
                    score_latent=z['score_logits'].reshape(-1),
                    movement_x=z['movement'][:,owners,0].reshape(-1),movement_y=z['movement'][:,owners,1].reshape(-1))
                table=pa.table(rows)
                if writer is None:writer=pq.ParquetWriter(target,table.schema,compression='zstd')
                writer.write_table(table)
    finally:
        if writer is not None:writer.close()
    return writer is not None


def verify(root):
    root=Path(root);lock=locked(root);assert not lock['smoke']
    assert file_hash(root/'configs/calibration_grid.csv')==lock['grid_sha256']
    audits=[];hashes={}
    for checkpoint in ('best','last'):
        for seed in ACTION_SEEDS:
            p=root/'s1'/checkpoint/str(seed)
            meta=json.loads((p/'provenance.json').read_text());assert (p/'complete.json').exists()
            assert meta['weights_unchanged'] and not meta['optimizer_created']
            assert meta['repeats']==16
            hashes[checkpoint]=meta['checkpoint_sha256']
            for regression in meta['replay_regression']:
                assert all(v<=1e-6 for v in regression['max_absolute_errors'].values())
                assert set(regression['max_absolute_errors'])=={'coverage','throughput_bps','reward','service_interruptions','direct_switches'}
            audits.extend(meta['replay_regression'])
            for split in ('development','confirmation'):
                for rule in (('R1','R2','R3') if split=='development' else ('R0','R1','R2','R3')):
                    case=root/'s2/rollout'/split/checkpoint/rule/str(seed)
                    contract=json.loads((case/'complete.json').read_text())
                    assert contract['checkpoint_sha256']==hashes[checkpoint]
                    assert contract['horizon']==400 and contract['scenes']==(SCENES if split=='development' else CONFIRMATION)
                    assert contract['beta']==lock['beta'] and contract['load_weight']==lock['load_weight']
                    m=json.loads((case/'provenance.json').read_text());assert m['weights_unchanged'] and not m['optimizer_created']
    # External UE streams must match in all closed-loop rules, including baselines.
    for ck in ('best','last'):
        for seed in ACTION_SEEDS:
            for split in ('development','confirmation'):
                digest=[]
                for rule in (('R1','R2','R3') if split=='development' else ('R0','R1','R2','R3')):
                    p=root/'s2/rollout'/split/ck/rule/str(seed)/'provenance.json'
                    digest.append(json.loads(p.read_text())['ue_path_sha256'])
                if split=='development':
                    p=root/'s1'/ck/str(seed)/'provenance.json'
                    digest.append(json.loads(p.read_text())['replay_regression'][0]['ue_path_sha256'])
                assert len(set(digest))==1
    # A fresh actor-driven R0 rollout (not only recorded-motion replay) must
    # reproduce old T2. This checks that the monotone score transform never
    # perturbed the actor observation, random stream, or subsequent motion.
    fresh=[]
    for ck in ('best','last'):
        new=root/'s2/rollout/development'/ck/'R0/880001'
        old=resolve_output(REFERENCE)/'rollout'/ck/'T2/880001'
        if (old/'closed_loop').exists():
            assert (new/'complete.json').exists()
            block_hashes=[]
            for p in sorted((old/'closed_loop').glob('*.npz')):
                q=new/'closed_loop'/p.name
                with np.load(p) as a,np.load(q) as b:
                    for field in ('movement','uav_xy','user_xy','association','proposals','B'):
                        np.testing.assert_array_equal(a[field],b[field])
                block_hashes.append(dict(old=str(p),new=str(q),old_sha256=file_hash(p),new_sha256=file_hash(q)))
            assert len(block_hashes)==16
            with np.load(old/'step_metrics.npz') as a,np.load(new/'step_metrics.npz') as b:
                errors={k:float(np.max(np.abs(a['metrics'][:,:,a['names'].tolist().index(k)]-
                    b['metrics'][:,:,b['names'].tolist().index(k)]))) for k in
                    ('coverage','throughput_bps','reward','service_interruptions','direct_switches')}
            assert max(errors.values())<=1e-6
            fresh.append(dict(checkpoint=ck,actor_seed=880001,steps=12800,max_errors=errors,blocks=block_hashes))
    if fresh:atomic_json(root/'audit/fresh_r0_regression.json',dict(status='passed',cases=fresh))
    else:
        saved=json.loads((root/'audit/fresh_r0_regression.json').read_text())
        assert saved['status']=='passed' and len(saved['cases'])==2
    atomic_json(root/'audit/baseline_regression.json',dict(status='passed',model_hashes=hashes,
        baseline_replays=audits,all_six_reference_paths_exact=True,weights_unchanged=True,optimizer_created=False,
        rule_lock_sha256=lock['contract_sha256'],confirmation_new_scenes=CONFIRMATION,
        development_new_transitions=230400,confirmation_new_transitions=307200,calibration_transitions=12800,
        fresh_actor_r0_regression='audit/fresh_r0_regression.json',
        note='No learner/critic/optimizer instantiated. Physical exogenous UE paths agree across rules.'))
    print('S1/S2 frozen-source and trajectory verification passed',flush=True)


def figures(root,rank,noise,effects,events,survival,summary,paired,series):
    import matplotlib.pyplot as plt
    from .plotting import style,save_figure
    style();out=root/'figures'
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ck,col in [('best','#009E73'),('last','#0072B2')]:
        rr=[r for r in rank if r['checkpoint']==ck and r['pool']=='fixed' and r['stratum'] in ('le10','11to15','gt15')]
        nn=[r for r in noise if r['checkpoint']==ck and r['pool']=='fixed' and r['stratum'] in ('le10','11to15','gt15')]
        for ax,key,data in [(axes[0],'spearman',rr),(axes[1],'topk_jaccard',rr),(axes[2],'pairwise_topk_jaccard',nn)]:
            labels={'le10':'≤10','11to15':'11–15','gt15':'>15'}
            ax.plot([labels[r['stratum']] for r in data],[r.get(key,np.nan) for r in data],
                marker='o' if ck=='best' else '^',ls='-' if ck=='best' else '--',label=ck,color=col)
            ylabel={'spearman':'Learned-score / SINR Spearman',
                'topk_jaccard':'Learned / SINR Top-10 Jaccard',
                'pairwise_topk_jaccard':'Repeated-sample Top-10 Jaccard'}[key]
            ax.set(xlabel='Candidate count',ylabel=ylabel)
    axes[0].set_ylim(-1,1)
    axes[1].set_ylim(0,1.03);axes[2].set_ylim(0,1.03)
    axes[0].legend();save_figure(fig,out,'ranking_topk_noise')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    causecolors=['#333333','#0072B2','#56B4E9','#E69F00','#009E73','#999999']
    for ax,ck in zip(axes,('best','last')):
        bottom=np.zeros(2)
        for cause,color in zip(CAUSES,causecolors):
            vals=[next(r['per_episode_last100'] for r in events if r['checkpoint']==ck and r['protocol']==p and r['cause']==cause) for p in ('T1','T2')]
            ax.bar(['Learned','SINR'],vals,bottom=bottom,label=cause,color=color);bottom+=vals
        ax.set(title=ck,ylabel='Interruptions per episode, steps 301-400')
    axes[1].legend(fontsize=6);save_figure(fig,out,'interruption_causes')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,ck in zip(axes,('best','last')):
        rr=[r for r in effects if r['checkpoint']==ck]
        for r in rr:
            ax.scatter(r['coverage']*100,r['throughput_bps']/1e6,color=COLORS[r['score_mode']],s=50,label=r['score_mode'])
            ax.annotate(r['score_mode'],(r['coverage']*100,r['throughput_bps']/1e6),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(title=ck,xlabel='Fixed-motion coverage (%)',ylabel='Throughput (Mbps)')
    save_figure(fig,out,'score_mean_vs_single')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ck,ls in [('best','-'),('last','--')]:
        for protocol,col in [('T1','#009E73'),('T2','#0072B2')]:
            rr=[r for r in survival if r['checkpoint']==ck and r['protocol']==protocol]
            axes[0].step([r['duration'] for r in rr],[r['survival'] for r in rr],where='post',color=col,ls=ls,label=ck+'/'+protocol)
    axes[0].set(xlabel='Service-spell duration (steps)',ylabel='KM survival',xlim=(0,150));axes[0].legend(fontsize=7)
    loads=[numeric(r) for r in read_csv(root/'s1/load_structure.csv')]
    for protocol,col in [('T1','#009E73'),('T2','#0072B2')]:
        rr=[r for r in loads if r['protocol']==protocol]
        axes[1].plot(range(11),[np.mean([r[f'load_{n}']/18 for r in rr]) for n in range(11)],marker='o',color=col,label=protocol)
    axes[1].set(xlabel='Current served UEs per UAV',ylabel='Fraction of UAV states');axes[1].legend()
    save_figure(fig,out,'service_survival_and_load')
    for split in ('development','confirmation'):
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for i,ck in enumerate(('best','last')):
            for rule in ('T1','R0','R1','R2','R3'):
                hit=[v for (sp,c,r,s),v in series.items() if (sp,c,r)==(split,ck,rule)]
                if not hit:continue
                for ax,key,scale in [(axes[i,0],'coverage',100),(axes[i,1],'throughput_bps',1e-6)]:
                    y=np.mean([x[:,:,names.index(key)].mean(0) for x,names in hit],0)
                    ax.plot(np.arange(1,401),y*scale,label=rule,color=COLORS[rule],ls=STYLES[rule])
                    ax.set(xlabel='Executed step',ylabel='Coverage (%)' if key=='coverage' else 'Throughput (Mbps)',title=ck)
            axes[i,0].legend(ncol=3,fontsize=7)
        save_figure(fig,out,f'closure_curves_{split}')
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        for ax,ck in zip(axes,('best','last')):
            for row in summary:
                if (row['split'],row['checkpoint'])!=(split,ck):continue
                ax.scatter(row['last100_coverage']*100,row['last100_throughput_mbps'],color=COLORS[row['rule']],s=45)
                ax.annotate(row['rule'],(row['last100_coverage']*100,row['last100_throughput_mbps']),xytext=(3,4),textcoords='offset points')
            ax.set(title=ck,xlabel='Last-100 coverage (%)',ylabel='Last-100 throughput (Mbps)')
        save_figure(fig,out,'rule_tradeoffs_'+split)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,ck in zip(axes,('best','last')):
        for split,color,shift in [('development','#009E73',-.12),('confirmation','#0072B2',.12)]:
            for i,metric in enumerate(('last100_coverage','last100_throughput_mbps')):
                row=next(r for r in paired if r['split']==split and r['checkpoint']==ck and r['comparison']=='interaction' and r['metric']==metric)
                scale=100 if i==0 else 1
                x=row['difference']*scale
                ax.errorbar(i+shift,x,yerr=[[max(0,x-row['low']*scale)],[max(0,row['high']*scale-x)]],fmt='o',color=color,label=split if i==0 else None)
        ax.axhline(0,color='k',lw=.8);ax.set(title=ck,xticks=[0,1],xticklabels=['Coverage (pp)','Throughput (Mbps)'],ylabel='2 x 2 interaction')
        ax.legend(fontsize=7)
    save_figure(fig,out,'rule_interactions')


def markdown(root,rank,noise,effects,paired_noise,event_summary,summary,paired):
    lock=locked(root)
    lines=['# S1／S2：UE评分诊断与物理规则消融','',
        '**已完成冻结诊断，不包含任何新训练。** A3 best来自251,904步，last来自1,000,000步；两个检查点分别报告，不重新选模。',
        f'独立校准场景740000–740015锁定β={lock["beta"]}、λ={lock["load_weight"]}；开发验证10000–10031；新确认场景750000–750031。每组3个动作种子880001–880003、400步、批次16。',
        '所有对照均保持运动前评分／Top10提议→一次运动→后状态过滤→UE选择，不补位、不引入新候选。奖励仍为覆盖＋0.1×归一化吞吐。',
        '区间为场景级配对bootstrap（2000次、950001）：先平均同场景的3个动作种子，再重抽32个场景；不是训练种子置信区间。','',
        '## S1：排序与采样','',
        '主排序统计仅使用固定时点50/100/150/200/250/300/325/350/375/400的动作前观察；例如step50对应执行第50次动作之前的t=49。每场景每拥挤层另抽一个时点，保存纳入概率和逆概率权重；额外样本单独展示，不混入主表。','',
        '| 模型 | 候选层 | 有定义Spearman数 | Spearman | 学习/SINR Top-K Jaccard | 多次采样 Top-K Jaccard | 第10/11名翻转率 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for ck in ('best','last'):
        for st in ('all','crowded','11to15','gt15'):
            rr=next(r for r in rank if r['checkpoint']==ck and r['pool']=='fixed' and r['stratum']==st)
            nn=next(r for r in noise if r['checkpoint']==ck and r['pool']=='fixed' and r['stratum']==st)
            fmt=lambda v:'—' if v is None else f'{v:.4f}'
            lines.append(f'| {ck} | {st} | {rr["defined_correlations"]} | {fmt(rr["spearman"])} | {fmt(rr["topk_jaccard"])} | {fmt(nn["pairwise_topk_jaccard"])} | {fmt(nn["boundary_rank_flip"])} |')
    lines+=['','| 模型 | 固定运动评分 | 即时覆盖 | 即时吞吐 Mbps | 中断数/步 | 跨UAV最优链路选择一致率 |','|---|---|---:|---:|---:|---:|']
    for r in effects:
        lines.append(f'| {r["checkpoint"]} | {r["score_mode"]} | {r["coverage"]:.4%} | {r["throughput_bps"]/1e6:.3f} | {r["service_interruptions"]:.3f} | {r["best_rate_choice_agreement"]:.2%} |')
    lines+=['','16次采样保存的是同一观察下完整联合动作的最终运动、评分及latent；影子关联始终使用原轨迹的同一次物理运动和后状态。这里不是“给定运动后的条件评分分布”。mean16是实际sigmoid评分均值，额外16次推理成本单列。',
        '逐UE评分与运动可从分块NPZ重建；本地额外导出Parquet。组内Spearman不等于跨UAV分数可比性；跨UAV有效提议比较另列在score_effect_summary.csv。','',
        '## S1：服务中断归因','',
        '| 模型 | 评分 | 后100步中断/回合 | 回传失效 | 链路/发现失效 | 前置候选缺失 | Top-K未提议 | 竞争未接纳 | 其他 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for ck in ('best','last'):
        for p in ('T1','T2'):
            rr=[r for r in event_summary if (r['checkpoint'],r['protocol'])==(ck,p)]
            row=[next(r for r in rr if r['cause']==cause) for cause in CAUSES]
            lines.append('| '+ ' | '.join([ck,p,f'{sum(r["per_episode_last100"] for r in row):.3f}']+[f'{r["fraction"]:.2%}' for r in row])+' |')
    lines+=['','主原因按回传→链路/发现→前置候选→Top-K→竞争→其他互斥分类，同时保存重叠标签。若存在任何有效提议，UE必然被服务；因此本协议下“仅因跨UAV选择竞争导致完全中断”应为0。直接换UAV仍有服务，另计切换，不算中断。',
        '服务段指同一UAV连续服务同一UE，直接切换会结束旧连接段，但不算完全服务中断。已结束与400步仍活动的右删失样本分别保存，生存曲线使用Kaplan–Meier；初始关联从reset起计可观察服务时间，不推断仿真之前的时长。','',
        '## S2：闭环表现','',
        'Q=ln(1+10^(SINR/10))/ln(11)，R0=Q，R1=Q+β×既有连接，R2=Q−λ×前负载/10，R3两项叠加。未做UAV内归一化。λ对本机候选是常数，只影响跨UAV选择，不改变其Top-K。',
        'β、λ从预注册网格{0.05,0.15,0.30}正值组合中，按独立校准轨迹的平均即时任务奖励选择；没有用开发/确认结果调参。该校准准则不保证长期最优，失败也不否定所有权重。','',
        '| 集合 | 模型 | 规则 | 后100覆盖 | 后50覆盖 | 终态覆盖 | 后100吞吐Mbps | 后100σ pp | 后100中断/回合 | 直接切换/回合 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["split"]} | {r["checkpoint"]} | {r["rule"]} | {r["last100_coverage"]:.4%} | {r["last50_coverage"]:.4%} | {r["final_coverage"]:.4%} | {r["last100_throughput_mbps"]:.3f} | {r["last100_coverage_std_pp"]:.3f} | {r["last100_service_interruptions_window_sum"]:.3f} | {r["last100_direct_switches_window_sum"]:.3f} |')
    lines+=['','| 集合 | 模型 | 差值 | 后100覆盖差 pp [95%区间] | 吞吐差 Mbps [95%区间] | 中断差/回合 [95%区间] |','|---|---|---|---:|---:|---:|']
    for split in ('development','confirmation'):
        for ck in ('best','last'):
            for comparison in ('R1 - R0','R2 - R0','R3 - R0','interaction'):
                values=[]
                for metric,scale in [('last100_coverage',100),('last100_throughput_mbps',1),('last100_service_interruptions_window_sum',1)]:
                    r=next(x for x in paired if (x['split'],x['checkpoint'],x['comparison'],x['metric'])==(split,ck,comparison,metric))
                    values.append(f'{r["difference"]*scale:+.3f} [{r["low"]*scale:+.3f}, {r["high"]*scale:+.3f}]')
                lines.append('| '+' | '.join([split,ck,comparison]+values)+' |')
    lines+=['','## 诊断结论','']
    for ck in ('best','last'):
        rank_row=next(r for r in rank if (r['checkpoint'],r['pool'],r['stratum'])==(ck,'fixed','crowded'))
        nn=next(r for r in noise if (r['checkpoint'],r['pool'],r['stratum'])==(ck,'fixed','crowded'))
        e=next(r for r in event_summary if (r['checkpoint'],r['protocol'],r['cause'])==(ck,'T1','not_in_topk'))
        rates={r['score_mode']:r for r in effects if r['checkpoint']==ck}
        gain_mean=(rates['mean16']['throughput_bps']-rates['recorded']['throughput_bps'])/1e6
        gain_sinr=(rates['sinr']['throughput_bps']-rates['recorded']['throughput_bps'])/1e6
        lines.append(f'- **{ck}**：拥挤集Spearman={rank_row["spearman"]:.4f}；重复采样Top-K Jaccard={nn["pairwise_topk_jaccard"]:.4f}，第10/11名翻转率={nn["boundary_rank_flip"]:.2%}。后100步学习评分中断的{e["fraction"]:.2%}主要归于Top-K未提议。')
        lines.append(f'  在原运动固定时，16次评分均值相对原单次评分的吞吐差为{gain_mean:+.3f} Mbps，而SINR为{gain_sinr:+.3f} Mbps。该对照同时检查噪声和排序；若均值收益仍远小于SINR，则仅收紧采样并未恢复物理链路排序。')
    lines+=['','原始候选≤10时，所有候选都能进入本机Top-K，因此全体Jaccard较高不能证明评分学得好；必须看拥挤层及跨UAV归属。服务保持项主要尝试缓解边界名次翻转，负载项只可能改变UE跨UAV选择，其作用空间受到多候选竞争比例限制。',
        '总名额18×10=180，大于150名UE，即便全覆盖也至少有30个空闲名额；不能把空闲率本身称为容量浪费。当前每UE固定2 MHz与功率，减轻实际负载不会自动增加单UE带宽。',
        '规则的即时校准目标与闭环长期目标不同。以上配对表若出现中断减少但吞吐/覆盖下降，应判为取舍，不能宣布规则全面改进；新确认集是锁定权重后的检查，不继续调参。',
        '后续训练侧诊断应优先检查评分的物理排序监督是否保留、评分分支的优势/梯度信号及与运动的耦合。S1/S2本身没有训练，也不能凭这些结果确定唯一的网络或PPO错误。']
    coverage_checks=[r for r in paired if r['comparison'] in ('R1 - R0','R2 - R0','R3 - R0') and r['metric']=='last100_coverage']
    lines.append(f'\n开发与确认、best与last的{len(coverage_checks)}项规则覆盖对照中，区间下界大于0的有{sum(r["low"]>0 for r in coverage_checks)}项；不能据此认定保持/负载规则稳定优于纯SINR。')
    lines.append('\n上述中断归因是UE级事件，包含小规模反复丢服务；不等于全部≥10pp的整体覆盖骤降，也不证明所有部署维持问题都由评分造成。')
    benchmark=root/'audit/inference_benchmark.json'
    if benchmark.exists():
        bench=json.loads(benchmark.read_text())
        lines+=['','## 推理成本','',
            '同一固定观察批次（16环境、288个UAV局部样本），预热后计时；并发负载可能影响绝对值，不含物理计算和磁盘。','',
            '| 模型 | 方法 | 批次中位耗时 ms |','|---|---|---:|']
        for r in bench['rows']:lines.append(f'| {r["checkpoint"]} | {r["method"]} | {r["median_seconds"]*1000:.3f} |')
    lines+=['','交互项为(R3−R2)−(R1−R0)。主要结论看覆盖、吞吐和中断共同变化，不能仅按切换次数挑规则。各闭环的后续运动允许因关联反馈改变，不是同一状态的单步因果效应。',
        '开发集T1/R0复用原始轨迹并完整重放校验；S2仅新增R1/R2/R3。确认集另执行R0/R1/R2/R3，不接触保留测试集。',
        '全部统计与图表可由保存数据重建；完整每步CSV为s2/per_step.csv，跨组主表为s2/comparison.csv，区间为s2/paired_differences.csv。',
        '这里只定位评分排序、随机采样及局部规则的作用，不能据此证明DPPO概率、网络结构或信用分配的唯一根因。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'s1/report.md').write_text('\n'.join(lines[:lines.index('## S2：闭环表现')])+'\n')
    (root/'s2/report.md').write_text('# S2：规则闭环消融\n\n'+'\n'.join(lines[lines.index('## S2：闭环表现'):])+'\n')
    (root/'README.md').write_text('# S1/S2诊断结果\n\n[完整报告](report.md)；[S1](s1/report.md)；[S2](s2/report.md)。\n\n主表：s2/comparison.csv。所有动作重复先按场景合并；置信区间不代表训练种子。\n')


def rebuild(output):
    root=Path(output);verify(root)
    rank,noise,effects,noise_pairs,events,survival=s1_tables(root)
    summary,paired,series=s2_tables(root)
    figures(root,rank,noise,effects,events,survival,summary,paired,series)
    markdown(root,rank,noise,effects,noise_pairs,events,summary,paired)
    parquet=export_parquet(root)
    atomic_json(root/'audit/rebuild.json',dict(status='complete',parquet=parquet,
        comparison_rows=len(summary),paired_rows=len(paired),figures=len(list((root/'figures').glob('*.png')))))
    print('rebuilt S1/S2',root,flush=True)
