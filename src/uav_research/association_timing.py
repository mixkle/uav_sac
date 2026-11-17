"""Evaluation-only motion/association timing adapter; production step is unchanged."""
import copy
import numpy as np
from .diffusion_dppo.environment import ServiceEnvironment, resolve_topk

PROTOCOLS = ('T1', 'T2', 'P1', 'P2', 'T3')


def sinr_scores(y, threshold):
    # Exact F2b/D3 mapping. Never normalize separately within each UAV.
    return np.exp(-np.logaddexp(0., -(np.asarray(y)-threshold)/10.))


def solutions(before, after, learned, pre_scores, post_scores, capacity):
    shape = before.shape
    if (before.dtype != np.bool_ or after.dtype != np.bool_ or after.shape != shape
            or any(x.shape != shape or not np.isfinite(x).all()
                   for x in (learned, pre_scores, post_scores)) or capacity < 1):
        raise ValueError('Invalid timing candidates/scores/capacity')
    out = {}
    out['T1'] = resolve_topk(before, after, learned, capacity)
    out['T2'] = resolve_topk(before, after, pre_scores, capacity)
    # Valid old proposals already contain at most K entries per UAV. Their
    # support remains fixed: only the UE choice is evaluated using new scores.
    valid = out['T2'][2]
    c, _, _ = resolve_topk(valid, valid, post_scores, capacity)
    out['P1'] = (c, out['T2'][1].copy(), valid.copy())
    out['P2'] = resolve_topk(before & after, after, post_scores, capacity)
    out['T3'] = resolve_topk(after, after, post_scores, capacity)
    for c, _, _ in out.values():
        assert (c.sum(0) <= 1).all() and (c.sum(1) <= capacity).all()
        assert not ((c != 0) & ~after).any()
    return out


def outcome(c, proposals, valid, rates, previous, ages, last_owner, users, capacity):
    owner = np.where(c.any(0), c.argmax(0), -1)
    served = owner >= 0
    same = served & (owner == previous)
    ended = (previous >= 0) & ~same & (ages > 0)
    new_age = np.where(same, ages+1, served.astype(np.int64))
    coverage, rate = float(c.sum()/users), float((c*rates).sum())
    return dict(coverage=coverage, throughput_bps=rate, reward=coverage+.1*rate/(users*5e6),
        proposal_count=int(proposals.sum()), invalidated_proposals=int((proposals & ~valid).sum()),
        duplicate_proposals=int(valid.sum()-c.sum()), topk_unused_slots=int(capacity*len(c)-c.sum()),
        direct_switches=int((served & (previous >= 0) & ~same).sum()),
        service_interruptions=int(((previous >= 0) & ~served).sum()),
        service_reconnections=int((served & (previous < 0) & (last_owner >= 0)).sum()),
        completed_service_spells=int(ended.sum()),
        completed_service_duration_sum=int(ages[ended].sum()),
        service_age_mean=float(new_age[served].mean()) if served.any() else 0.)


class TimingEnvironment(ServiceEnvironment):
    def __init__(self, cfg, protocol='T1', shadow=True):
        if protocol not in ('T1', 'T2', 'T3'):
            raise ValueError('Only T1/T2/T3 can be executed')
        self.protocol, self.shadow = protocol, shadow
        super().__init__(cfg)

    def prepare(self, action):
        # Production validation still checks IDs, finite scores and disk movement.
        self.pre_y = self.Y.copy()
        self.pre_ages = self.service_age.copy()
        self.pre_last_owner = self.last_owner.copy()
        self.pre_positions = (self.uav_xy.copy(), self.user_xy.copy())
        return super().prepare(action)

    def finish(self, prepared, links):
        before, learned, previous = prepared
        _, after = self._graph(links, None)
        pre = sinr_scores(self.pre_y, self.cfg.sinr_threshold_db)
        post = sinr_scores(links[2], self.cfg.sinr_threshold_db)
        if self.shadow:
            solved = solutions(before, after, learned, pre, post, self.cfg.capacity)
            self.shadow_rows = {name: outcome(*parts, links[3], previous, self.pre_ages,
                self.pre_last_owner, self.cfg.num_users, self.cfg.capacity) for name, parts in solved.items()}
            self.shadow_associations = {name: parts[0].astype(np.uint8) for name, parts in solved.items()}
        else:
            self.shadow_rows = {}
        effective = prepared if self.protocol == 'T1' else (
            (before, pre, previous) if self.protocol == 'T2' else (after, post, previous))
        # Service age / last owner / graph / observations are committed ONCE.
        result = super().finish(effective, links)
        if self.shadow:
            np.testing.assert_array_equal(self.C, solved[self.protocol][0])
            for k in ('coverage', 'throughput_bps', 'reward', 'direct_switches',
                      'service_interruptions', 'service_reconnections', 'service_age_mean'):
                np.testing.assert_allclose(self.info[k], self.shadow_rows[self.protocol][k], rtol=1e-12, atol=1e-9)
        self.timing = dict(candidate_before_edges=int(before.sum()), candidate_after_edges=int(after.sum()),
            new_candidate_edges=int((after & ~before).sum()), lost_candidate_edges=int((before & ~after).sum()),
            new_candidate_unique_ues=int((after & ~before).any(0).sum()))
        self.timing_before, self.timing_after = before.copy(), after.copy()
        return result

    def snapshot(self):
        return dict(super().snapshot(), timing_protocol=self.protocol)

    def restore(self, state):
        if state.get('timing_protocol', self.protocol) != self.protocol:
            raise ValueError('Timing protocol snapshot mismatch')
        super().restore(state)
