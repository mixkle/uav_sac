"""Optional follow-up parameterizations; legacy A3/A1 actor stays untouched."""
import torch
from torch import nn
from ..system_ablation.policy import AblationDiffusion
from ..system_ablation.settings import options
from ..diffusion_dppo.policy import mean_segments


class FollowupDiffusion(AblationDiffusion):
    def __init__(self,cfg):
        super().__init__(cfg)
        if options(cfg)['diffusion_init_mode']=='normal_reference_residual':
            for head in (self.movement_head,self.score_head):
                if isinstance(head,nn.Identity):continue  # Motion-only actor has no score parameters.
                nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def _noise_residual(self,enc,movement,scores,step):
        if not options(self.cfg)['disable_noisy_score_to_motion']:
            return super().predict_noise(enc,movement,scores,step)
        own=enc.owners
        pairs=self.pair(torch.cat([enc.candidates,scores[:,None]],-1))
        pooled=mean_segments(pairs,enc.lengths)
        time=self.time_embedding[step]
        if time.ndim==1:time=time.expand(len(movement),-1)
        mov=self.movement_head(torch.cat([movement,enc.context,torch.zeros_like(pooled),time],-1))
        # Scoring retains the original noisy pair pool and all contexts.
        score=self.score_head(torch.cat([scores[:,None],enc.candidates,
            enc.context[own],movement[own],pooled[own],time[own]],-1)).squeeze(-1)
        return mov,score

    def predict_noise(self,enc,movement,scores,step):
        mov,score=self._noise_residual(enc,movement,scores,step)
        if options(self.cfg)['diffusion_init_mode']=='normal_reference_residual':
            b=self.sqrt_one_minus_abar[step]
            if b.ndim:return mov+b[:,None]*movement,score+b[enc.owners]*scores
            return mov+b*movement,score+b*scores
        return mov,score

    def transition(self,enc,movement,scores,step):
        if options(self.cfg)['diffusion_init_mode']!='normal_reference_residual':
            return super().transition(enc,movement,scores,step)
        # Equivalent DDPM posterior without z-b*b*z cancellation at high noise.
        # Sampling and probability recomputation share this exact same method.
        dm,ds=self._noise_residual(enc,movement,scores,step)
        a,b=self.sqrt_abar[step],self.sqrt_one_minus_abar[step]
        c0,ct=self.coef_x0[step],self.coef_xt[step]
        keep,gain=c0*a+ct,c0*b/a
        if keep.ndim:
            return keep[:,None]*movement-gain[:,None]*dm,keep[enc.owners]*scores-gain[enc.owners]*ds
        return keep*movement-gain*dm,keep*scores-gain*ds
