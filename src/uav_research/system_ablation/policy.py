"""Actor variants; old Gaussian/DDPM factories retain their original path."""
import numpy as np
import torch
from torch import nn
from ..diffusion_dppo.gaussian import GaussianServiceActor
from ..diffusion_dppo.policy import DiffusionActor, network, mean_segments
from ..diffusion_dppo.observations import ServiceObservation, ServiceAction, offsets
from .settings import options
from .encoding import setup, encode



class MotionOnly:
    def sample(self, obs, generator=None, deterministic=False):
        actual = obs
        action,chain = super().sample(obs,generator=generator,deterministic=deterministic)
        if not options(self.cfg)['score_actions']:
            logits=(actual.users[actual.service_mask,3]*self.cfg.env.sinr_scale_db-self.cfg.env.sinr_threshold_db)/10.
            scores=np.exp(-np.logaddexp(0.,-logits)).astype(np.float32)
            action=ServiceAction(action.movement,scores,actual.candidate_ids.copy(),offsets(actual.candidate_lengths))
        return action,chain

    def encode(self, obs):
        enc=encode(self,obs)
        if not options(self.cfg)['score_actions']:
            enc.candidates=enc.candidates[:0]
            enc.lengths=torch.zeros_like(enc.lengths)
            enc.owners=enc.owners[:0]
        return enc


class AblationGaussian(MotionOnly,GaussianServiceActor):
    def __init__(self,cfg):
        super().__init__(cfg)
        setup(self,cfg)
        if not options(cfg)['score_actions']:
            self.score_head=nn.Identity()

    def distribution(self, enc):
        if options(self.cfg)['score_actions']:
            pooled=mean_segments(enc.candidates,enc.lengths)
            s=self.score_head(torch.cat([enc.candidates,enc.context[enc.owners],pooled[enc.owners]],-1))
            # Only movement context is limited in C; scoring sees all candidates.
            m=self.movement_head(torch.cat([enc.context,enc.motion_pool],-1))
            return (m[:,:2],m[:,2:].clamp(self.cfg.algo.log_std_min,self.cfg.algo.log_std_max),
                    s[:,0],s[:,1].clamp(self.cfg.algo.log_std_min,self.cfg.algo.log_std_max))
        m=self.movement_head(torch.cat([enc.context,enc.motion_pool],-1))
        empty=m.new_empty(0)
        return m[:,:2],m[:,2:].clamp(self.cfg.algo.log_std_min,self.cfg.algo.log_std_max),empty,empty

    def denoising_loss(self,obs,target_movement,target_scores,**unused):
        """BC is joint latent NLL, split into movement/score coordinate sums."""
        from ..diffusion_dppo.policy import sum_segments
        import math
        enc=self.encode(obs.tensor(next(self.parameters()).device))
        mm,ml,sm,sl=self.distribution(enc)
        m=torch.as_tensor(target_movement,device=mm.device)
        s=torch.as_tensor(target_scores,device=mm.device)
        mov=(.5*((m-mm)*(-ml).exp()).square()+ml+.5*math.log(2*math.pi)).sum(-1)
        ls=.5*((s-sm)*(-sl).exp()).square()+sl+.5*math.log(2*math.pi)
        score=sum_segments(ls[:,None],enc.lengths).squeeze(-1)
        return mov,score


class AblationDiffusion(MotionOnly,DiffusionActor):
    def __init__(self,cfg):
        super().__init__(cfg)
        setup(self,cfg)
        if not options(cfg)['score_actions']:
            self.pair=nn.Identity()
            self.score_head=nn.Identity()
            self.movement_head=network(2+cfg.diffusion.context_width+cfg.diffusion.time_width,256,2)

    def predict_noise(self,enc,movement,scores,step):
        if options(self.cfg)['score_actions']: return super().predict_noise(enc,movement,scores,step)
        time=self.time_embedding[step]
        if time.ndim==1: time=time.expand(len(movement),-1)
        return self.movement_head(torch.cat([movement,enc.context,time],-1)),movement.new_empty(0)

    def denoising_loss(self,obs,target_movement,target_scores,**kwargs):
        if not options(self.cfg)['score_actions']:
            target_scores=np.empty(0,np.float32)
            if kwargs.get('noise') is not None:
                kwargs['noise']=(kwargs['noise'][0],kwargs['noise'][1][:0])
        return super().denoising_loss(obs,target_movement,target_scores,**kwargs)


def make_actor(cfg):
    o=options(cfg)
    if o['disable_noisy_score_to_motion'] or o['diffusion_init_mode']!='epsilon':
        from ..diffusion_followup.policy import FollowupDiffusion
        return FollowupDiffusion(cfg)
    return (AblationGaussian if cfg.algo.name=='gaussian_service_mappo' else AblationDiffusion)(cfg)
