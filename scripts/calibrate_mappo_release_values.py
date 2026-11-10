"""Frozen-actor MC calibration at the release of 5/20-step interventions.

Reconstruct endpoints, never resample MC futures. Split whole original scenes.
Fit absolute MC labels; select on paired release-value-difference RMSE.
No actor, GAE, PopArt, online checkpoint, or production algorithm is changed.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import yaml

from check_mappo_action_blocks import adapt_case, block_action
from check_mappo_action_values import FIELDS, MODES, SOURCE, HISTORY, RUNS, module_hashes
from compare_online_time_factor import write_csv
from report_mappo_bridge import digest
from uav_research.config import load_config
from uav_research.evaluate import policy_from_checkpoint
from uav_research.observations import stack_observations
from uav_research.plotting import COLORS, plt, save_figure, style
from uav_research.runtime import atomic_json, atomic_save, configure_runtime, isolated_rng
from uav_research.vector import VectorEnvironment


GROUPS = [(150,5),(150,20),(200,5),(200,20)]
KIND = 'offline_release_mc_critic_v1'


def scene_split(seeds, cfg):
    seeds=np.asarray(seeds)
    counts=[cfg[k] for k in ['fit_scenes','selection_scenes','check_scenes']]
    if len(np.unique(seeds))!=len(seeds) or sum(counts)!=len(seeds) or min(counts)<1:
        raise ValueError('Need distinct scene seeds and an exhaustive nonempty split')
    order=np.random.default_rng(cfg['split_seed']).permutation(len(seeds))
    edges=np.cumsum([0]+counts)
    return {name:np.sort(order[edges[i]:edges[i+1]]) for i,name in enumerate(['fit','selection','check'])}


def mc_tail(reward, gamma, k):
    if not 0<=k<reward.shape[-1]:raise ValueError('Release must precede terminal state')
    carry=np.zeros(reward.shape[:-1],np.float64)
    for j in range(reward.shape[-1]-1,k-1,-1):carry=reward[...,j]+gamma*carry
    return carry


class ReleaseCritic(nn.Module):
    """An exact source time-factor critic copy with frozen source output units."""
    def __init__(self, policy):
        super().__init__()
        self.net=copy.deepcopy(policy.value).requires_grad_(True)
        self.horizon=copy.deepcopy(policy.horizon_factor)
        self.register_buffer('average_mean',policy.average_normalizer.mean.detach().clone())
        self.register_buffer('average_std',policy.average_normalizer.std.detach().clone())

    def forward(self, state):
        return self.horizon(state)*(self.net(state).squeeze(-1)*self.average_std+self.average_mean)


def fit_epoch(model, optimizer, states, targets, scale, rng, cfg):
    model.train();order=rng.permutation(len(states));losses=[];norms=[]
    for start in range(0,len(order),cfg['batch_size']):
        ix=order[start:start+cfg['batch_size']]
        x=torch.from_numpy(np.array(states[ix],dtype=np.float32,copy=True))
        y=torch.as_tensor(targets[ix],dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(x)
        loss=nn.functional.huber_loss(prediction/scale,y/scale,delta=cfg['huber_delta'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MC calibration loss')
        loss.backward()
        norm=nn.utils.clip_grad_norm_(model.net.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()));norms.append(float(norm))
    return float(np.mean(losses)),float(np.mean(norms))


@torch.no_grad()
def predict(model, states, batch=256):
    model.eval();shape=states.shape[:-1];flat=states.reshape(-1,states.shape[-1]);parts=[]
    for j in range(0,len(flat),batch):
        parts.append(model(torch.from_numpy(np.array(flat[j:j+batch],copy=True))).numpy())
    result=np.concatenate(parts).reshape(shape).astype(np.float64)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite calibrated values')
    return result


def paired_error(prediction, target):
    """[group, mode, repeat, scene]; policy reference appears once per group."""
    if prediction.shape!=target.shape or prediction.ndim!=4:raise ValueError('Expected aligned grouped values')
    return (prediction[:,1:]-prediction[:,:1])-(target[:,1:]-target[:,:1])


def scores(prediction,target):
    error=prediction-target;pair=paired_error(prediction,target)
    return dict(rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.abs(error).mean()),
                bias=float(error.mean()),pair_rmse=float(np.sqrt(np.mean(pair**2))),
                pair_mae=float(np.abs(pair).mean()),pair_bias=float(pair.mean()))


def save_critic(path, model, optimizer, epoch, rng, protocol, scale, source_hash):
    atomic_save(path,dict(kind=KIND,model=model.state_dict(),optimizer=optimizer.state_dict(),
        epoch=epoch,permutation_rng=copy.deepcopy(rng.bit_generator.state),protocol=protocol,
        fixed_mc_scale=scale,source_checkpoint_sha256=source_hash,
        output_semantics='H(t) * frozen_source_average_units(MLP(state)); cumulative full environment reward',
        online_resume_allowed=False))


def restore_critic(path, model, optimizer, rng):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('kind')!=KIND:raise ValueError('Not a release-MC offline checkpoint')
    model.load_state_dict(data['model']);optimizer.load_state_dict(data['optimizer'])
    rng.bit_generator.state=data['permutation_rng']
    return data


def reconstruct(cfg, policy, case, mode, repeat, length, saved):
    """Exactly replay the saved prefix; no additional future is sampled."""
    snaps=copy.deepcopy(case['snapshots']);start=case['start'];end=start+len(snaps)
    if repeat:
        for snap,record in zip(snaps,case['records']):
            snap['movement_rng']=np.random.default_rng(np.random.SeedSequence(
                [982001,record['seed'],case['time'],repeat])).bit_generator.state
    mi=MODES.index(mode)
    with isolated_rng(),torch.inference_mode():
        if repeat==0:torch.set_rng_state(case['torch_rng'].clone())
        else:torch.manual_seed(int(np.random.SeedSequence(
            [981001,case['time'],case['start'],repeat]).generate_state(1)[0]))
        with VectorEnvironment(cfg.env,len(snaps),'batch') as vector:
            vector.restore(snaps);obs=stack_observations([e.obs for e in vector.envs])
            for j in range(length):
                raw=policy.act(obs,deterministic=False);action=block_action(raw,mode,j,length)
                if mode!='policy':
                    np.testing.assert_array_equal(raw,saved['prefix_raw'][mi-1,repeat,start:end,j])
                    np.testing.assert_array_equal(action,saved['prefix_effective'][mi-1,repeat,start:end,j])
                obs,state,_,done,trunc,infos=vector.step(action)
                assert not done.any() and not trunc.any()
                expected=saved['diagnostics'][mi,repeat,start:end,j]
                np.testing.assert_array_equal([[info[f] for f in FIELDS] for info in infos],expected)
            np.testing.assert_array_equal(policy.values(state).numpy(),saved['values'][mi,repeat,start:end,length])
            np.testing.assert_allclose(state[:,-1],(case['time']+length)/cfg.env.horizon,atol=1e-7)
            users=np.stack([e.user_xy for e in vector.envs])
    return state.copy(),users


def collect(out, protocol, meta):
    source=Path(protocol['source'])
    parent=json.loads((source/'experiment.json').read_text());assert parent['status']=='completed'
    cases=torch.load(SOURCE/'branch_states.pt',map_location='cpu',weights_only=False)
    histories=torch.load(HISTORY/'branch_histories.pt',map_location='cpu',weights_only=False)
    for label,run in RUNS.items():
        cfg,policy=policy_from_checkpoint(run/'last.pt','cpu');configure_runtime(cfg)
        policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
        before=module_hashes(policy);assert before==parent['groups'][label]['module_hashes']
        folder=out/label;folder.mkdir(exist_ok=True)
        states=np.lib.format.open_memmap(folder/'states.npy',mode='w+',dtype=np.float32,
            shape=(4,3,8,32,cfg.env.state_dim))
        target=np.empty((4,3,8,32));prefix=np.empty_like(target);initial_mc=np.empty_like(target)
        old=np.empty_like(target);good=[]
        for gi,(at,k) in enumerate(GROUPS):
            with np.load(source/label/f'K{k}/t{at}/curves.npz') as data:
                saved={key:data[key].copy() for key in data.files}
            gamma=float(saved['discount']);assert gamma==cfg.algo.discount
            r=saved['diagnostics'][...,FIELDS.index('reward')]
            np.testing.assert_allclose(saved['contributions'].sum(-1),r,atol=1e-12,rtol=1e-12)
            target[gi]=mc_tail(r,gamma,k);prefix[gi]=(r[...,:k]*gamma**np.arange(k)).sum(-1)
            initial_mc[gi]=mc_tail(r,gamma,0);old[gi]=saved['values'][...,k]
            np.testing.assert_allclose(initial_mc[gi],prefix[gi]+gamma**k*target[gi],atol=1e-10)
            good.append([record['good'] for record in parent['records'][str(at)]])
            for case in [adapt_case(c,cfg,histories) for c in cases if c['time']==at]:
                start=case['start'];end=start+len(case['snapshots'])
                for repeat in range(8):
                    users_reference=None
                    for mi,mode in enumerate(MODES):
                        state,users=reconstruct(cfg,policy,case,mode,repeat,k,saved)
                        states[gi,mi,repeat,start:end]=state
                        if users_reference is None:users_reference=users
                        else:np.testing.assert_array_equal(users,users_reference)
                    meta['reconstruction_checks']+=3
            states.flush()
            print(f'{label}: reconstructed t{at}+K{k}; exact commands, radio diagnostics, value and UE pairing',flush=True)
        assert module_hashes(policy)==before
        assert all(p.grad is None for p in policy.actor.parameters())
        np.savez_compressed(folder/'labels.npz',returns=target,prefix=prefix,initial_mc=initial_mc,
            source_values=old,good=np.array(good),scene_seeds=saved['scene_seeds'],gamma=gamma,
            groups=np.array(GROUPS),modes=np.array(MODES))
        meta['models'][label]=dict(source_module_hashes=before,state_shape=list(states.shape))
        del states
        atomic_json(out/'experiment.json',meta)


def train(out,protocol,meta):
    for label,run in RUNS.items():
        folder=out/label;states=np.load(folder/'states.npy',mmap_mode='r')
        with np.load(folder/'labels.npz') as labels:
            target=labels['returns'];source_values=labels['source_values'];seeds=labels['scene_seeds']
        splits=scene_split(seeds,protocol);fit=splits['fit'];selection=splits['selection']
        cfg,policy=policy_from_checkpoint(run/'last.pt','cpu');configure_runtime(cfg)
        policy.actor.eval().requires_grad_(False);policy.value.eval().requires_grad_(False)
        source_hashes=module_hashes(policy);model=ReleaseCritic(policy)
        np.testing.assert_array_equal(predict(model,states),source_values)
        fit_x=states[:,:,:,fit,:].reshape(-1,states.shape[-1]).copy()
        fit_y=target[:,:,:,fit].reshape(-1).copy()
        select_x=states[:,:,:,selection,:].copy();select_y=target[:,:,:,selection]
        scale=max(float(fit_y.std(ddof=0)),1e-6)
        optimizer=torch.optim.Adam(model.net.parameters(),lr=protocol['learning_rate'],eps=protocol['adam_eps'])
        assert not optimizer.state
        rng=np.random.default_rng(protocol['permutation_seed']);logs=[]
        best=float('inf');best_epoch=0
        predictions={'source':source_values}
        bias=float((fit_y-source_values[:,:,:,fit].reshape(-1)).mean())
        predictions['fit_bias_only']=source_values+bias
        def measure(epoch, loss=None, grad=None):
            nonlocal best,best_epoch
            fit_pred=predict(model,fit_x)
            val_pred=predict(model,select_x)
            fs=scores(fit_pred.reshape(4,3,8,len(fit)),target[:,:,:,fit])
            vs=scores(val_pred,select_y)
            row=dict(epoch=epoch,loss=loss,gradient_norm=grad,**{f'fit_{k}':v for k,v in fs.items()},
                     **{f'selection_{k}':v for k,v in vs.items()})
            logs.append(row)
            if vs['pair_rmse']<best:
                best=vs['pair_rmse'];best_epoch=epoch
                save_critic(folder/'best.pt',model,optimizer,epoch,rng,protocol,scale,digest(run/'last.pt'))
            return row
        measure(0)
        started=time.perf_counter()
        try:
            for epoch in range(1,protocol['epochs']+1):
                loss,grad=fit_epoch(model,optimizer,fit_x,fit_y,scale,rng,protocol)
                row=measure(epoch,loss,grad)
                if epoch%25==0:
                    print(f'{label} epoch {epoch}: fit RMSE {row["fit_rmse"]:.3f}, selection pair RMSE {row["selection_pair_rmse"]:.3f}, best {best_epoch} ({time.perf_counter()-started:.1f}s)',flush=True)
                    write_csv(folder/'learning.csv',logs)
                    save_critic(folder/'progress.pt',model,optimizer,epoch,rng,protocol,scale,digest(run/'last.pt'))
        except (FloatingPointError,RuntimeError):
            save_critic(folder/'failure.pt',model,optimizer,epoch,rng,protocol,scale,digest(run/'last.pt'))
            write_csv(folder/'learning.csv',logs)
            raise
        save_critic(folder/'last.pt',model,optimizer,epoch,rng,protocol,scale,digest(run/'last.pt'))
        predictions['last']=predict(model,states)
        restore_critic(folder/'best.pt',model,optimizer,rng)
        predictions['best']=predict(model,states)
        np.savez_compressed(folder/'predictions.npz',**predictions)
        write_csv(folder/'learning.csv',logs)
        assert module_hashes(policy)==source_hashes
        assert all(p.grad is None for p in policy.actor.parameters())
        meta['models'][label].update(best_epoch=best_epoch,fixed_loss_scale=scale,fit_only_bias=bias,
            elapsed_fit_seconds=time.perf_counter()-started,
            split_seeds={key:seeds[ix].tolist() for key,ix in splits.items()},
            source_modules_unchanged=True,actor_gradient_absent=True)
        atomic_json(out/'experiment.json',meta)


def error_interval(source_error, calibrated_error, seed, repeats):
    """Scene-clustered RMSE change. Repeat and comparison axes remain grouped."""
    a=np.mean(source_error**2,axis=tuple(range(source_error.ndim-1)))
    b=np.mean(calibrated_error**2,axis=tuple(range(calibrated_error.ndim-1)))
    draws=np.random.default_rng(seed).integers(len(a),size=(repeats,len(a)))
    delta=np.sqrt(b[draws].mean(1))-np.sqrt(a[draws].mean(1))
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(delta=float(np.sqrt(b.mean())-np.sqrt(a.mean())),low=float(lo),high=float(hi))


def publish(out):
    meta=json.loads((out/'experiment.json').read_text());protocol=meta['protocol']
    summary=[];detail=[];changes=[]
    style()
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    scatter,sa=plt.subplots(2,2,figsize=(10,8))
    means,ma=plt.subplots(2,4,figsize=(14,7))
    for li,label in enumerate(RUNS):
        folder=out/label
        with np.load(folder/'labels.npz') as d:labels={k:d[k].copy() for k in d.files}
        with np.load(folder/'predictions.npz') as d:preds={k:d[k].copy() for k in d.files}
        target=labels['returns'];splits=scene_split(labels['scene_seeds'],protocol)
        for split,ix in splits.items():
            for ck,pred in preds.items():
                summary.append(dict(model=label,split=split,checkpoint=ck,scenes=len(ix),
                                    **scores(pred[:,:,:,ix],target[:,:,:,ix])))
        check=splits['check'];pair_target=target[:,1:]-target[:,:1]
        for ck in ['best','last']:
            for metric,old,new in [
                ('absolute_rmse',preds['source']-target,preds[ck]-target),
                ('pair_rmse',paired_error(preds['source'],target),paired_error(preds[ck],target))]:
                changes.append(dict(model=label,checkpoint=ck,metric=metric,scenes=len(check),
                    **error_interval(old[...,check],new[...,check],protocol['bootstrap_seed'],protocol['bootstrap_replicates'])))
        for gi,(at,k) in enumerate(GROUPS):
            for mi,mode in enumerate(MODES[1:],1):
                for split,ix in splits.items():
                    for subset in ['all','good']:
                        selected=ix if subset=='all' else ix[labels['good'][gi,ix]]
                        if not len(selected):continue
                        truth=(target[gi,mi]-target[gi,0])[:,selected]
                        prefix=(labels['prefix'][gi,mi]-labels['prefix'][gi,0])[:,selected]
                        for ck,pred in preds.items():
                            estimate=(pred[gi,mi]-pred[gi,0])[:,selected]
                            bootstrap=prefix+float(labels['gamma'])**k*estimate
                            full_mc=prefix+float(labels['gamma'])**k*truth
                            np.testing.assert_allclose(bootstrap-full_mc,float(labels['gamma'])**k*(estimate-truth),atol=1e-11)
                            scene_truth=truth.mean(0);scene_pred=estimate.mean(0)
                            detail.append(dict(model=label,time=at,length=k,mode=mode,split=split,subset=subset,
                                checkpoint=ck,scenes=len(selected),tail_mc_delta=float(truth.mean()),
                                tail_predicted_delta=float(estimate.mean()),mc_delta=float(full_mc.mean()),
                                bootstrap_delta=float(bootstrap.mean()),pair_rmse=float(np.sqrt(np.mean((estimate-truth)**2))),
                                scene_mean_pair_rmse=float(np.sqrt(np.mean((scene_pred-scene_truth)**2))),
                                sign_agreement=float(np.mean(np.sign(scene_truth)==np.sign(scene_pred)))))
            # Unchanged, previously defined good subset within the calibration check split.
            ix=check[labels['good'][gi,check]]
            ax=ma[li,gi];x=np.arange(2);width=.25
            for ci,ck in enumerate(['source','best','last']):
                p=preds[ck][gi,1:]-preds[ck][gi,:1]
                ax.bar(x+(ci-1)*width,p[...,ix].mean((1,2)),width,label=ck,color=COLORS[ci])
            ax.plot(x,pair_target[gi][:,:,ix].mean((1,2)),'kx',ms=8,label='MC')
            ax.axhline(0,color='gray',lw=.7);ax.set(xticks=x,xticklabels=['Half','Hover'],
                title=f'{label}: t{at}, K{k}, good n={len(ix)}',ylabel='Release value difference')
            if gi==0:ax.legend(fontsize=6)
        import csv
        with (folder/'learning.csv').open() as f:rows=list(csv.DictReader(f))
        epochs=[int(r['epoch']) for r in rows]
        for col,key in enumerate(['rmse','pair_rmse']):
            for split in ['fit','selection']:
                axes[li,col].plot(epochs,[float(r[f'{split}_{key}']) for r in rows],label=split)
            axes[li,col].axvline(meta['models'][label]['best_epoch'],color='gray',ls=':',label='selected best')
            axes[li,col].set(title=f'{label}: {key}',xlabel='Offline epoch',ylabel='Full reward units');axes[li,col].legend()
        for col,ck in enumerate(['source','best']):
            truth=pair_target[...,check].mean(2).ravel()
            pred=(preds[ck][:,1:]-preds[ck][:,:1])[...,check].mean(2).ravel()
            sa[li,col].scatter(truth,pred,s=15,alpha=.65)
            low=min(truth.min(),pred.min());high=max(truth.max(),pred.max())
            sa[li,col].plot([low,high],[low,high],'--',color='gray')
            sa[li,col].set(title=f'{label}: {ck}, check scenes',xlabel='MC tail difference (8 futures mean)',ylabel='Predicted release value difference')
    save_figure(fig,out/'figures','learning_curves')
    save_figure(scatter,out/'figures','paired_value_scatter')
    save_figure(means,out/'figures','good_release_value_differences')
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_details.csv',detail);write_csv(out/'check_changes.csv',changes)
    report=['# 保持结束状态的 MC critic 校准','',
        '冻结原 actor，复用 K5/K20 的 MC 尾部，只拟合结束状态。原 time-factor MLP 全参数可训练；原输出单位和两套在线归一化统计不更新。',
        '按场景分为16拟合、8选模、8检查，所有时刻/分支/未来重复随场景归组。检查场景没有参与本次拟合、损失尺度估计或选模，但这32个场景已用于此前研究诊断，不能称为全新测试集。','',
        f'300轮，batch256，Adam lr5e-4/eps1e-5，重置Adam，梯度裁剪0.5；固定拟合集MC总体标准差缩放后的Huber(delta10)。无GAE/bootstrap/value clipping/actor更新。每轮遍历全部拟合状态，按选模集配对价值差RMSE选择best（含epoch0）。','',
        '|模型|检查集版本|绝对RMSE|配对价值差RMSE|配对MAE|', '|---|---|---:|---:|---:|']
    for row in summary:
        if row['split']=='check':report.append(f'|{row["model"]}|{row["checkpoint"]}|{row["rmse"]:.3f}|{row["pair_rmse"]:.3f}|{row["pair_mae"]:.3f}|')
    report+=['','配对误差定义为 `(V_hold−V_policy)−(MC_hold−MC_policy)`，同时包括半幅和悬停。policy端点在训练中只保存一次，不因比较对象重复加权。','',
        '|模型|版本|检查集指标变化（新−原）|场景bootstrap 95%区间|','|---|---|---|---|']
    for row in changes:
        report.append(f'|{row["model"]}|{row["checkpoint"]}|{row["metric"]}: {row["delta"]:+.3f}|[{row["low"]:+.3f}, {row["high"]:+.3f}]|')
    report+=['','区间为2000次场景级重采样；同一场景的时刻/模式/未来重复完整保留，不是训练种子区间。MC每个端点只有一个已保存未来实现，标签带噪声。',
        '仅增加拟合集平均误差的常数修正作为参考：可以修正绝对偏差，但理论上不能改变同一起点的价值差。','',
        '![学习曲线](figures/learning_curves.png)','![配对价值差](figures/paired_value_scatter.png)',
        '![原good子集的释放状态比较](figures/good_release_value_differences.png)','',
        '各t/K/模式及good/all子集的MC与bootstrap见 `paired_details.csv`。逐状态标签、原预测、best/last预测保存在各模型目录，`states.npy`为重建的6292维物理状态。',
        '本实验不改变动作轨迹，覆盖率也不会因此提高；即便校准有效，也不能据此声称在线actor已获得更好的更新信号。历史actor依赖一步历史，但critic仍只读物理状态，这一限制保持原样。',
        '校准仅覆盖t155/170/205/220附近的干预状态；没有证明整个回合的价值都准确，也没有单独区分新数据分布与移除bootstrap的贡献。',
        '离线检查点只允许本脚本恢复，不能用作在线MAPPO续训。正式模型、actor、优化器与原始数据均保留。','']
    (out/'report.md').write_text('\n'.join(report))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/critic_release_mc_calibration.yaml')
    parser.add_argument('--rebuild',action='store_true');args=parser.parse_args()
    protocol=yaml.safe_load(Path(args.config).read_text());out=Path(protocol['output'])
    if args.rebuild:publish(out);return
    cfg=load_config(RUNS['M1-C']/'config.yaml');configure_runtime(cfg)
    assert torch.get_num_threads()==8
    out.mkdir(parents=True,exist_ok=False)
    source=Path(protocol['source'])
    inputs=[source/'experiment.json',SOURCE/'branch_states.pt',HISTORY/'branch_histories.pt',
            Path(args.config),Path(__file__),Path('scripts/check_mappo_action_blocks.py')]
    inputs += [run/name for run in RUNS.values() for name in ['last.pt','best.pt','config.yaml']]
    inputs += [source/label/f'K{k}/t{at}/curves.npz' for label in RUNS for at,k in GROUPS]
    inputs += list(Path('src/uav_research').glob('*.py'))
    hashes={str(p):digest(p) for p in inputs};started=time.perf_counter()
    meta=dict(status='running',protocol=protocol,source_sha256=hashes,models={},reconstruction_checks=0,
              actor_updates=0,cpu_threads=8,numba_threads=4,selection_metric='paired_release_value_difference_rmse')
    atomic_json(out/'experiment.json',meta)
    collect(out,protocol,meta);train(out,protocol,meta)
    assert hashes=={str(p):digest(p) for p in inputs}
    meta.update(status='completed',source_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    atomic_json(out/'experiment.json',meta);publish(out)


if __name__=='__main__':main()
