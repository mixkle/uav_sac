"""Read-only source reproduction and paired-effect acceptance checks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from uav_research.association_timing_experiment import ACTION_SEEDS
from uav_research.runtime import resolve_output,atomic_json
from uav_research.topk_diagnostic import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='analysis/t1_t2_t3_post_motion_association')
    a=p.parse_args();root=resolve_output(a.output)
    checks=[]
    keys=['coverage','throughput_bps','reward','disconnection_ratio','discovery_fraction',
          'direct_switches','service_interruptions','service_reconnections','service_age_mean']
    for checkpoint in ('best','last'):
        source=resolve_output(f'runs/system_ablation_1m/A3_seed0/{checkpoint}.pt')
        for seed in ACTION_SEEDS:
            paths=[]
            for protocol in ('T1','T2','T3'):
                case=root/'rollout'/checkpoint/protocol/str(seed)
                meta=json.loads((case/'provenance.json').read_text())
                assert json.loads((case/'complete.json').read_text())['horizon']==400
                assert meta['checkpoint_sha256']==file_hash(source)
                assert meta['weights_unchanged'] and not meta['optimizer_created']
                assert meta['scene_seeds']==list(range(10000,10032))
                assert meta['peak_rss_gib']<=12
                paths.append(meta['ue_path_sha256'])
            assert len(set(paths))==1
            for protocol,old in [
                ('T1',resolve_output(f'analysis/diffusion_gaussian_ablation_1m/A3/validation/{checkpoint}/{seed}')),
                ('T2',resolve_output(f'runs/diffusion_followup_1m/audits/F2b_frozen/sinr/{checkpoint}/{seed}'))]:
                case=root/'rollout'/checkpoint/protocol/str(seed)
                with np.load(old/'step_metrics.npz') as z, np.load(case/'step_metrics.npz') as now:
                    old_names=z['names'].tolist();new_names=now['names'].tolist()
                    np.testing.assert_array_equal(z['scene_seeds'],now['scene_seeds'])
                    errors={}
                    for key in keys:
                        x=z['metrics'][:,:,old_names.index(key)];y=now['metrics'][:,:,new_names.index(key)]
                        errors[key]=float(np.abs(x-y).max())
                        np.testing.assert_allclose(x,y,rtol=0,atol=1e-6)
                checks.append(dict(checkpoint=checkpoint,protocol=protocol,action_seed=seed,
                                   reproduced=old.as_posix(),max_absolute_errors=errors))
    # Each row shares one physical transition, previous owner, service age and
    # last-owner array; differences must telescope before any aggregation.
    telescope={};min_rate_gain=np.inf
    for checkpoint in ('best','last'):
        for seed in ACTION_SEEDS:
            path=root/'rollout'/checkpoint/'T1'/str(seed)/'paired_step_metrics.csv'
            by_state={}
            with path.open() as f:
                for row in csv.DictReader(f):
                    by_state.setdefault((int(row['scene_seed']),int(row['step'])),{})[row['protocol']]=row
            assert len(by_state)==32*400
            for state,rows in by_state.items():
                assert set(rows)=={'T1','T2','P1','P2','T3'}
                assert float(rows['P1']['coverage'])==float(rows['T2']['coverage'])
                min_rate_gain=min(min_rate_gain,float(rows['P1']['throughput_bps'])-float(rows['T2']['throughput_bps']))
                for key in ('coverage','throughput_bps','reward','direct_switches','service_interruptions'):
                    q=[float(rows[p][key]) for p in ('T2','P1','P2','T3')]
                    error=abs((q[1]-q[0])+(q[2]-q[1])+(q[3]-q[2])-(q[3]-q[0]))
                    telescope[key]=max(telescope.get(key,0),error)
                    assert error<2e-6
    assert min_rate_gain>=-2e-6
    atomic_json(root/'tests/acceptance.json',dict(passed=True,reproduction=checks,
        shadow_states=2*3*32*400,telescoping_max_error=telescope,
        p1_coverage_identical_to_t2=True,p1_minimum_throughput_gain_bps=float(min_rate_gain),
        ue_paths_identical=True,source_files_unchanged=True,no_optimizers=True))
    print('PASS: 18 frozen cases, 12 historical reproductions, 76,800 same-state shadows.',flush=True)


if __name__=='__main__':main()
