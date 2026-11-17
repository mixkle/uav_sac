"""New namespace: never overwrite the legacy F2b=A3 alias or frozen configs."""
import copy
import hashlib
import json
from pathlib import Path
import yaml
from uav_research.config import load_config

ROOT=Path(__file__).resolve().parents[1]
DATA='/mnt/disk2/dym/uav_sac'


def build():
    target=ROOT/'configs/diffusion_followup_1m'
    target.mkdir(exist_ok=True)
    entries={}
    for group,base in [('F2b','A3'),('F2c','A3'),('A1b','A1')]:
        cfg=copy.deepcopy(load_config(ROOT/f'configs/system_ablation_1m/{base}.yaml').as_dict())
        o=cfg['diffusion']['ablation']
        o.update(run_id=group,disable_noisy_score_to_motion=(group=='F2c'),
                 diffusion_init_mode='normal_reference_residual' if group=='A1b' else 'epsilon')
        if group=='F2b':
            o.update(association='sinr_topk',initialization='actor',
                     initial_checkpoint=DATA+'/runs/system_ablation_1m/A3_seed0/step0.pt')
        elif group=='F2c':
            o['initial_checkpoint']=DATA+'/runs/diffusion_followup_1m/pretraining/F2c/best.pt'
        cfg['diffusion']['actor_update_devices']=[]
        path=target/f'{group}.yaml'
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        load_config(path)
        entries[group]=dict(config=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=DATA+f'/runs/diffusion_followup_1m/{group}_seed0',reference=base)
    manifest=dict(version='diffusion-followup-f2b-f2c-a1b-v1',runs=entries,
        old_alias_unchanged='configs/system_ablation_1m/manifest.json still maps F2b to A3; use ONLY follow-up CLI for new F2b',
        demonstrations=DATA+'/runs/system_ablation_1m/demonstrations',
        action_seeds=[880001,880002,880003],
        init_audit=dict(pools='16 original training worker reset scenes + 128 independent reset scenes 932000..932127',
            repeats=64,seed=993001,thresholds=dict(latent_norm_mean_min=.1,latent_norm_mean_max=5.,latent_norm_p99_max=15.,
                movement_norm_p95_max=.99,movement_norm_mean_max=.95,score_saturation_01_max=.05,
                movement_norm_mean_min=.1,score_latent_std_min=.05,
                logp_abs_tolerance=2e-3,logp_relative_tolerance=2e-4)),
        sensitivity_seed=993101,
        training=dict(online_joint_steps=1_000_000,seed=0,pretrain_F2c_epochs=50,
            F2b_actor='exact A3 online step0 actor; fresh critic/normalizers/Adam',
            F2c_actor='same data/random seed, gate disabled during ALL BC/online/inference',
            A1b_actor='no teacher/BC; normal-reference epsilon residual, zero output projections'))
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
