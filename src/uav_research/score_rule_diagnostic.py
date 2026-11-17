"""Frozen, evaluation-only UE score diagnostics. No production policy changes."""
import numpy as np
from .diffusion_dppo.environment import ServiceEnvironment, resolve_topk

CAUSES = ('backhaul', 'link_or_discovery', 'pre_candidate_missing',
          'not_in_topk', 'competition_not_accepted', 'other')


def quality(sinr_db):
    """A globally comparable, stable spectral-efficiency score, Q(10 dB)=1."""
    return np.logaddexp(0., np.asarray(sinr_db, np.float64)*np.log(10)/10)/np.log(11)


def rule_scores(sinr_db, previous_c, capacity, beta=0., load_weight=0.):
    if capacity < 1 or min(beta, load_weight) < 0:
        raise ValueError('Invalid rule weights/capacity')
    c = np.asarray(previous_c)
    if c.shape != np.shape(sinr_db) or (c.sum(0) > 1).any():
        raise ValueError('Association and quality must align')
    return quality(sinr_db) + beta*c - load_weight*c.sum(1)[:, None]/capacity


def ranks(x):
    """Average ranks for ties; no scipy dependency on the training server."""
    x = np.asarray(x)
    order = np.argsort(x, kind='stable'); result = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        result[order[start:end]] = (start+end-1)/2
        start = end
    return result


def spearman(a, b):
    if len(a) < 2:
        return None
    a, b = ranks(a), ranks(b)
    a, b = a-a.mean(), b-b.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm else None


def chosen(ids, scores, capacity):
    ids = np.asarray(ids)
    return ids[np.lexsort((ids, -np.asarray(scores)))[:capacity]]


def ranking_rows(before, learned, sinr, post_rates, after, capacity):
    rows = []
    for u in range(len(before)):
        ids = np.flatnonzero(before[u]); n = len(ids)
        a = set(chosen(ids, learned[u, ids], capacity))
        b = set(chosen(ids, sinr[u, ids], capacity))
        rows.append(dict(uav=u, candidates=n, stratum=('le10' if n<=10 else '11to15' if n<=15 else 'gt15'),
            spearman=spearman(learned[u,ids],sinr[u,ids]),
            rate_spearman=spearman(learned[u,ids],post_rates[u,ids]),
            topk_jaccard=len(a&b)/len(a|b) if a|b else None,
            topk_overlap=len(a&b), high_quality_missed=len(b-a),
            learned_post_valid=sum(after[u,j] for j in a),
            quality_post_valid=sum(after[u,j] for j in b),
            usable_alternative_users=sum(after[u,j] for j in b-a)))
    return rows


def competition_metrics(valid, scores, rates):
    correlated=[]; agree=[]
    for m in np.flatnonzero(valid.sum(0)>1):
        ids=np.flatnonzero(valid[:,m]); s=scores[ids,m]; r=rates[ids,m]
        c=spearman(s,r)
        if c is not None: correlated.append(c)
        agree.append(int(ids[np.argmax(s)]==ids[np.argmax(r)]))
    return dict(competing_users=len(agree), competing_rank_count=len(correlated),
        competing_rank_sum=float(sum(correlated)), competing_best_rate_agreements=int(sum(agree)))


def interruption_rows(previous, current, before, after, proposals, hops_after,
                      sinr_before, sinr_after, ages, capacity):
    """Exclusive ordered causes AND overlapping evidence for every loss of service."""
    prev_owner=np.where(previous.any(0), previous.argmax(0),-1)
    owner=np.where(current.any(0),current.argmax(0),-1)
    rows=[]
    for m in np.flatnonzero((prev_owner>=0)&(owner<0)):
        i=int(prev_owner[m]); backhaul=bool(hops_after[i].min()>len(before))
        link=bool(not after[i,m]); missing=bool(not before[i,m]); topk=bool(not proposals[i,m])
        # With this one-round protocol, any valid proposal guarantees that UE
        # is served somewhere. Pure UE-choice competition cannot cause a loss.
        valid_old=bool(proposals[i,m] and after[i,m])
        if valid_old:
            raise AssertionError('Valid proposal left an unserved UE')
        flags=(backhaul,link,missing,topk,False,True)
        cause=next(c for c,f in zip(CAUSES,flags) if f)
        rows.append(dict(ue=int(m),old_uav=i,cause=cause,backhaul_failed=int(backhaul),
            link_or_discovery_failed=int(link and not backhaul),post_infeasible=int(link),
            pre_candidate_missing=int(missing),not_in_topk=int(topk),valid_old_proposal=int(valid_old),
            alternative_valid_proposals=int((proposals[:,m]&after[:,m]).sum()),
            old_candidate_count=int(before[i].sum()),old_load=int(previous[i].sum()),
            capacity=capacity,service_age=int(ages[m]),sinr_before_db=float(sinr_before[i,m]),
            sinr_after_db=float(sinr_after[i,m]),sinr_change_db=float(sinr_after[i,m]-sinr_before[i,m])))
    return rows


def survival_rows(durations, events):
    """Kaplan-Meier estimate, keeping active spells right censored."""
    durations, events=np.asarray(durations),np.asarray(events,bool)
    survival=1.; rows=[]
    for t in np.unique(durations):
        at_risk=int((durations>=t).sum()); died=int(((durations==t)&events).sum())
        censored=int(((durations==t)&~events).sum())
        survival*=1-died/at_risk
        rows.append(dict(duration=int(t),at_risk=at_risk,ended=died,censored=censored,survival=survival))
    return rows


class ScoreRuleEnvironment(ServiceEnvironment):
    """Frozen actor still samples the full action; only its executed scores differ."""
    def __init__(self, cfg, rule='T1', beta=0., load_weight=0.):
        if rule not in ('T1','R0','R1','R2','R3'):
            raise ValueError('Unknown score rule')
        self.rule,self.beta,self.load_weight=rule,beta,load_weight
        super().__init__(cfg)

    def prepare(self, action):
        self.pre_c=self.C.copy();self.pre_y=self.Y.copy();self.pre_age=self.service_age.copy()
        self.pre_last_owner=self.last_owner.copy()
        before,learned,previous=super().prepare(action)  # validate, then move exactly ONCE
        score = learned if self.rule=='T1' else rule_scores(self.pre_y,self.pre_c,self.cfg.capacity,
            self.beta if self.rule in ('R1','R3') else 0.,
            self.load_weight if self.rule in ('R2','R3') else 0.)
        self.diagnostic_before=before.copy()
        return before,score,previous

    def finish(self, prepared, links):
        _,after=self._graph(links,None)
        self.diagnostic_after=after.copy()
        result=super().finish(prepared,links)
        previous=self.pre_c; same=(previous.astype(bool)&self.C.astype(bool)).sum()
        rates=links[3][self.C.astype(bool)]
        self.extra=dict(link_retained=int(same),previous_links=int(previous.sum()),
            idle_slots=int(self.cfg.capacity*self.cfg.num_uavs-self.C.sum()),
            full_uav_fraction=float((self.counts==self.cfg.capacity).mean()),
            candidate_count_mean=float(prepared[0].sum(1).mean()),
            valid_proposal_multi_ue_fraction=float(((self.proposals&after).sum(0)>1).mean()),
            served_ue_rate_mean_mbps=float(rates.mean()/1e6) if len(rates) else 0.,
            served_ue_rate_p05_mbps=float(np.quantile(rates,.05)/1e6) if len(rates) else 0.,
            completed_service_duration_sum=self.info['completed_service_duration_sum'],
            completed_service_spells=self.info['completed_service_spells'])
        for n in range(self.cfg.capacity+1):
            self.extra[f'load_{n}_fraction']=float((previous.sum(1)==n).mean())
        result[-1].update(self.extra)
        return result
