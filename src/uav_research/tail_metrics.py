"""Post-deployment metrics from immutable full-episode evaluation arrays."""
from pathlib import Path
import numpy as np

METRICS=('tail100_coverage_pct','final_coverage_pct','tail100_throughput_mbps',
         'final_throughput_mbps','tail100_coverage_std_pp')


def read_series(path, *, throughput_scale_bps=None, coverage_name="coverage", throughput_name="throughput_bps"):
    """Return [scene,post-action time] service coverage and total throughput.

    Legacy reward curves store raw tau; its physical denominator must be supplied
    from the source configuration, never inferred from the weighted reward.
    """
    with np.load(Path(path),allow_pickle=False) as z:
        if 'diagnostics' in z:
            a=z['diagnostics'];names=z['diagnostic_names'].tolist()
        elif 'metrics' in z and 'names' in z:
            a=z['metrics'];names=z['names'].tolist()
        else:raise ValueError('No full episode scalar metric matrix')
        if a.ndim!=3 or coverage_name not in names:raise ValueError('Not scene/time/scalar metrics')
        coverage=np.asarray(a[:,:,names.index(coverage_name)],np.float64)
        if throughput_name in names:rate=np.asarray(a[:,:,names.index(throughput_name)],np.float64)
        elif 'raw' in z and 'raw_names' in z and throughput_scale_bps is not None:
            raw_names=z['raw_names'].tolist()
            if 'tau' not in raw_names:raise ValueError('No raw throughput label')
            rate=np.asarray(z['raw'][:,:,raw_names.index('tau')],np.float64)*throughput_scale_bps
        else:raise ValueError('Throughput series missing and no explicit tau scale')
        scene_key='scene_seeds' if 'scene_seeds' in z else 'seeds'
        if scene_key not in z:raise ValueError('Missing explicit scene IDs')
        scenes=np.asarray(z[scene_key],np.int64)
    validate_series(coverage,rate,scenes)
    return coverage,rate,scenes


def validate_series(coverage,rate,scenes):
    if coverage.ndim!=2 or coverage.shape!=rate.shape or coverage.shape[0]!=len(scenes):raise ValueError('Mismatched scene/time dimensions')
    if coverage.shape[1]<100:raise ValueError('Requires at least 100 post-action steps')
    if len(np.unique(scenes))!=len(scenes):raise ValueError('Duplicate scenes within one action repetition')
    if not np.isfinite(coverage).all() or not np.isfinite(rate).all():raise ValueError('Nonfinite metric')
    if np.any((coverage < -1e-9)|(coverage>1+1e-9)) or (rate < -1e-6).any():raise ValueError('Metric outside physical range')


def per_scene(coverage,rate,scenes):
    validate_series(coverage,rate,scenes)
    tail=coverage[:,-100:]
    return dict(scene_seed=scenes,tail100_coverage_pct=tail.mean(1)*100,
        final_coverage_pct=coverage[:,-1]*100,tail100_throughput_mbps=rate[:,-100:].mean(1)/1e6,
        final_throughput_mbps=rate[:,-1]/1e6,
        tail100_coverage_variance_pp2=tail.var(1,ddof=0)*10000,
        tail100_scene_std_pp=tail.std(1,ddof=0)*100,
        all_coverage_pct=coverage.mean(1)*100,all_throughput_mbps=rate.mean(1)/1e6)


def summarize(repetitions, bootstrap_repeats=2000, seed=950001):
    """Average per-trajectory statistics BEFORE repeats and scene bootstrap.

    Primary temporal SD=sqrt(mean within-trajectory population variance). Also
    report mean(scene SD) separately. Neither includes between-scene variation.
    """
    if not repetitions:raise ValueError('No repetitions')
    scenes=repetitions[0]['scene_seed']
    for r in repetitions:np.testing.assert_array_equal(r['scene_seed'],scenes)
    values={k:np.mean([r[k] for r in repetitions],axis=0) for k in repetitions[0] if k!='scene_seed'}
    n=len(scenes);idx=np.random.default_rng(seed).integers(n,size=(bootstrap_repeats,n))
    result={};intervals={}
    for key,a in values.items():
        result[key]=float(a.mean());intervals[key]=np.quantile(a[idx].mean(1),[.025,.975]).tolist()
    var=values['tail100_coverage_variance_pp2']
    result['tail100_coverage_std_pp']=float(np.sqrt(var.mean()))
    intervals['tail100_coverage_std_pp']=np.quantile(np.sqrt(var[idx].mean(1)),[.025,.975]).tolist()
    result['scene_count']=n;result['action_repetitions']=len(repetitions)
    return result,intervals,values
