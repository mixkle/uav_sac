"""Rebuild additional paired diagnostics from saved evaluation data only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from uav_research.plotting import style, save_figure
from uav_research.system_ablation.report import write_csv

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parents[1] / 'runs/diffusion_followup_1m/audits'


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def interval(values):
    values = np.asarray(values, float)
    draws = np.random.default_rng(950001).integers(len(values), size=(2000, len(values)))
    return [float(values.mean()), *np.quantile(values[draws].mean(1), [.025, .975]).tolist()]


def main():
    rows = read(ROOT / 'metrics/unified_step0_best_last.csv')
    episodes = read(ROOT / 'metrics/per_scene_action.csv')
    lookup = {(r['run_id'], r['checkpoint']): r for r in rows}
    groups = ['A1', 'A1b', 'G0', 'A3', 'F2b', 'F2c', 'F2a', 'A5']
    compact = []
    for g in groups:
        for kind in ('step0', 'best', 'last'):
            r = lookup[g, kind]
            compact.append(dict(group=g, checkpoint=kind, online_steps=r['online_steps'],
                coverage100_pct=100*float(r['last100_coverage']),
                coverage50_pct=100*float(r['last50_coverage']),
                final_coverage_pct=100*float(r['final_coverage']),
                throughput100_mbps=float(r['last100_throughput_mbps']),
                throughput50_mbps=float(r['last50_throughput_mbps']),
                final_throughput_mbps=float(r['final_throughput_mbps']),
                coverage100_std_pp=float(r['last100_coverage_std_pp']),
                drops100=float(r['last100_drop_events_10pp']),
                max_drop100_pp=float(r['last100_max_drop_pp']),
                connected100_pct=100*float(r['last100_connected_fraction']),
                p05_coverage100_pct=100*float(r['last100_coverage_p05']),
                maintain_pp=100*float(r['maintain_coverage']),
                return400=float(r['episode_return']),
                switches100=float(r['last100_direct_switches'])*100,
                interruptions100=float(r['last100_service_interruptions'])*100,
                actual_speed=float(r['last100_actual_speed_mean']),
                critic_rmse=float(r['critic_rmse_all'])))
    write_csv(ROOT/'metrics/concise_comparison.csv', compact)
    def scene(group, kind, key):
        source = [r for r in episodes if r['run_id']==group and r['checkpoint']==kind]
        return np.array([np.mean([float(r[key]) for r in source if int(r['scene_seed'])==s])
                         for s in range(10000,10032)])
    paired = []
    for left,right in [('F2b','A3'), ('F2c','A3'), ('A1b','A1'), ('A1b','G0'), ('F2a','F2b')]:
        for key in ['last100_coverage','last100_throughput_mbps','last100_service_interruptions',
                    'last100_drop_events_10pp','last100_coverage_std_pp','maintain_coverage']:
            raw = scene(left,'last',key)-scene(right,'last',key)
            gain = raw - (scene(left,'step0',key)-scene(right,'step0',key))
            d,lo,hi = interval(raw); dg,lg,hg=interval(gain)
            paired.append(dict(comparison=left+' - '+right, metric=key,
                last_difference=d,last_low=lo,last_high=hi,
                gain_difference=dg,gain_low=lg,gain_high=hg))
    write_csv(ROOT/'metrics/paired_last_and_learning_gain.csv',paired)
    gains=[]
    for g in groups:
        for key in ['last100_coverage','last100_throughput_mbps','maintain_coverage']:
            d,lo,hi=interval(scene(g,'last',key)-scene(g,'step0',key))
            gains.append(dict(group=g,metric=key,last_minus_step0=d,low=lo,high=hi))
    write_csv(ROOT/'metrics/paired_online_gains.csv',gains)
    summaries=[]
    for kind in ['step0','best','last']:
        p=AUDIT/'F2b_frozen/one_step'/kind/'step_protocol_diagnostics.csv'
        data=read(p)
        for label,predicate in [('all',lambda t:True),('last100',lambda t:t>=300)]:
            sub=[r for r in data if predicate(int(r['t']))]
            summaries.append(dict(checkpoint=kind,window=label,steps=len(sub),
                proposal_jaccard=np.mean([float(r['proposal_jaccard']) for r in sub]),
                owner_changed_fraction=np.mean([float(r['owner_changed_fraction']) for r in sub]),
                immediate_coverage_gain_pp=100*np.mean([float(r['sinr_coverage'])-float(r['learned_coverage']) for r in sub]),
                immediate_throughput_gain_mbps=np.mean([float(r['sinr_throughput_bps'])-float(r['learned_throughput_bps']) for r in sub])/1e6,
                immediate_interruptions_delta=np.mean([float(r['sinr_service_interruptions'])-float(r['learned_service_interruptions']) for r in sub])))
    write_csv(ROOT/'metrics/frozen_same_transition_summary.csv',summaries)
    style()
    colors=['#009E73','#0072B2','#E69F00','#333333']
    sets=[('score_execution',['A3','F2b','F2a']),('score_motion_coupling',['A3','F2c']),
          ('stable_initialization',['A1','A1b','G0'])]
    for name,selected in sets:
        fig,axs=plt.subplots(1,2,figsize=(10,3.8))
        for j,key in enumerate(['last100_coverage','last100_throughput_mbps']):
            for i,g in enumerate(selected):
                y=[float(lookup[g,k][key])*(100 if j==0 else 1) for k in ['step0','best','last']]
                axs[j].plot(range(3),y,marker=['o','s','^'][i],color=colors[i],
                            ls=['-','--','-.'][i],label=g)
            axs[j].set(xticks=range(3),xticklabels=['Step 0','Selected best','1M last'],
                       ylabel='Last-100 coverage (%)' if j==0 else 'Last-100 throughput (Mbps)')
        axs[0].legend();save_figure(fig,ROOT/'figures/interpretation',name+'_checkpoints')
    # Correct scope: fixed observation pools, not visited online trajectories.
    scales=[]
    for k in ['step0','audit_100k','best','last']:
        d=json.loads((ROOT/'A1b_scale_progress'/k/'summary.json').read_text())
        scales.append(dict(checkpoint=k,online_steps=d['online_steps'],**d['pools']['-1']))
    write_csv(ROOT/'metrics/A1b_fixed_pool_scale.csv',scales)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,label in zip(axs,['movement_latent_norm_mean','movement_norm_mean','score_saturation_01'],
                            ['Mean motion latent norm','Mean instruction norm','Score sigmoid saturation (<.01 or >.99)']):
        ax.plot(range(4),[s[key] for s in scales],color=colors[1],marker='o')
        ax.set(xticks=range(4),xticklabels=['Step0','100k','Best','Last'],ylabel=label)
    save_figure(fig,ROOT/'figures/interpretation','A1b_fixed_pool_scales')
    print(ROOT/'metrics/concise_comparison.csv')


if __name__=='__main__':
    main()
