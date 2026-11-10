"""Top-K service proposals with pre-move scores and post-move physical gating."""
import copy
import numpy as np
from numba import njit

from .. import kernels
from ..environment import UAVEnvironment, physical_links
from ..topk_diagnostic import _ranked_candidates
from .observations import ServiceObservation, ServiceAction, stack


@njit(cache=True)
def resolve_topk(before, after, scores, capacity):
    """Production P0: same ordering as the independent fixed-layout audit."""
    rank, lengths = _ranked_candidates(before, scores)
    u, m = before.shape
    proposals = np.zeros((u, m), np.bool_)
    for i in range(u):
        for k in range(min(capacity, lengths[i])):
            proposals[i, rank[i, k]] = True
    valid = proposals & after
    c = np.zeros((u, m), np.int64)
    for j in range(m):
        best = -1
        for i in range(u):
            if valid[i, j] and (best < 0 or scores[i, j] > scores[best, j]):
                best = i
        if best >= 0:
            c[best, j] = 1
    return c, proposals, valid


class ServiceEnvironment(UAVEnvironment):
    def reset(self, seed=None):
        self._initializing = True
        self.service_diagnostics = {}
        self.service_age = np.zeros(self.cfg.num_users, np.int64)
        self.last_owner = np.full(self.cfg.num_users, -1, np.int64)
        self.proposals = np.zeros((self.cfg.num_uavs, self.cfg.num_users), bool)
        self.last_scores = np.zeros_like(self.proposals, dtype=np.float64)
        try:
            result = super().reset(seed)
        finally:
            self._initializing = False
        self.last_owner = self.owner()
        return result

    def set_positions(self, *args, **kwargs):
        self._initializing = True
        self.service_diagnostics = {}
        self.service_age = np.zeros(self.cfg.num_users, np.int64)
        self.last_owner = np.full(self.cfg.num_users, -1, np.int64)
        self.proposals = np.zeros((self.cfg.num_uavs, self.cfg.num_users), bool)
        self.last_scores = np.zeros_like(self.proposals, dtype=np.float64)
        try:
            result = super().set_positions(*args, **kwargs)
        finally:
            self._initializing = False
        self.last_owner = self.owner()
        return result

    def owner(self):
        return np.where(self.C.any(0), self.C.argmax(0), -1)

    def _install(self, *args):
        super()._install(*args)
        self.legacy_obs = self.obs
        ids = np.concatenate([np.flatnonzero(row) for row in self.sensed]).astype(np.int64)
        group = np.repeat(np.arange(self.cfg.num_uavs), self.sensed.sum(1))
        old = self.obs.users
        users = np.column_stack([old[:, :3],
                 self.Y[group, ids] / self.cfg.sinr_scale_db, old[:, 3]]).astype(np.float32)
        feasible = ((self.Y[group, ids] >= self.cfg.sinr_threshold_db) &
                    (self.B.min(1)[group] <= self.cfg.num_uavs))
        self.obs = ServiceObservation(self.obs.fixed, users, self.obs.offsets,
                                      ids, feasible)
        self.info.update(getattr(self, "service_diagnostics", {}))
        # The legacy field also includes proposal/timing losses in this protocol.
        self.info["feasible_unserved_users"] = self.info["capacity_rejected_users"]

    def _graph(self, links, c):
        a, direct, y, rates, loss, uplink = links
        route = kernels.shortest_hops if self.cfg.backend == "numba" else kernels.shortest_hops.py_func
        failure = kernels.failure_coverage if self.cfg.backend == "numba" else kernels.failure_coverage.py_func
        label = kernels.component_labels if self.cfg.backend == "numba" else kernels.component_labels.py_func
        h = route(a, direct, np.ones(self.cfg.num_uavs, bool))
        feasible = ((uplink >= self.cfg.sensing_threshold_db) &
                    (y >= self.cfg.sinr_threshold_db) & (h.min(1)[:, None] <= self.cfg.num_uavs))
        if c is None:
            return h, feasible
        if np.any((c != 0) & ~feasible) or np.any(c.sum(0) > 1) or np.any(c.sum(1) > self.cfg.capacity):
            raise ValueError("Saved/installed association violates physical or capacity constraints")
        n = c.sum(1).astype(np.int64)
        retained, switched, post = failure(a, direct, n, h, self.cfg.num_users)
        return h, c, n, feasible.sum(1), retained, switched, post, label(a)

    def _refresh(self):
        if getattr(self, "_initializing", False):
            # Identical baseline initialization, then never greedily reassociate.
            return super()._refresh()
        links = physical_links(self.cfg, self.uav_xy, self.user_xy)
        zero_steps = self.zero_coverage_steps
        self._install(*links, self._graph(links, self.C.copy()))
        self.zero_coverage_steps = zero_steps
        self.info["zero_coverage_steps"] = zero_steps

    def prepare(self, action):
        if not isinstance(action, ServiceAction):
            raise ValueError("topk_score requires ServiceAction, not a movement-only array")
        o = self.obs
        expected_ids = o.candidate_ids
        expected_offsets = np.r_[0, np.cumsum(o.candidate_lengths)]
        if (not np.array_equal(action.offsets, expected_offsets) or
                not np.array_equal(action.ue_ids, expected_ids) or
                action.scores.shape != expected_ids.shape or
                not np.isfinite(action.scores).all() or
                np.any((action.scores < 0) | (action.scores > 1))):
            raise ValueError("Action scores/IDs/masks do not match current candidates")
        scores = np.zeros_like(self.Y, dtype=np.float64)
        before = np.zeros_like(self.sensed)
        for i in range(self.cfg.num_uavs):
            lo, hi = action.offsets[i:i+2]
            ids = action.ue_ids[lo:hi]
            scores[i, ids] = action.scores[lo:hi]
            before[i, ids] = True
        previous_owner = self.owner()
        self._move(action.movement)
        return before, scores, previous_owner

    def finish(self, prepared, links):
        before, scores, previous_owner = prepared
        h, after = self._graph(links, None)
        solver = resolve_topk if self.cfg.backend == "numba" else resolve_topk.py_func
        c, proposals, valid = solver(before, after, scores, self.cfg.capacity)
        owner = np.where(c.any(0), c.argmax(0), -1)
        served = owner >= 0
        same = served & (owner == previous_owner)
        reconnected = served & (previous_owner < 0) & (self.last_owner >= 0)
        reconnected_changed = reconnected & (owner != self.last_owner)
        ended = (previous_owner >= 0) & ~same & (self.service_age > 0)
        ended_ages = self.service_age[ended]
        self.service_age = np.where(same, self.service_age + 1, served.astype(np.int64))
        self.last_owner = np.where(served, owner, self.last_owner)
        self.service_diagnostics = dict(
            proposal_count=float(proposals.sum()),
            proposal_acceptance=float(c.sum() / max(1, proposals.sum())),
            invalidated_proposals=float((proposals & ~valid).sum()),
            duplicate_proposals=float(valid.sum() - c.sum()),
            unscored_waiting_edges=float((after & ~before).sum()),
            over_capacity_fraction=float((before.sum(1) > self.cfg.capacity).mean()),
            multi_candidate_fraction=float((before.sum(0) >= 2).mean()),
            direct_switches=float((served & (previous_owner >= 0) & ~same).sum()),
            service_interruptions=float(((previous_owner >= 0) & ~served).sum()),
            service_reconnections=float(reconnected.sum()),
            service_reconnections_changed_uav=float(reconnected_changed.sum()),
            completed_service_spells=float(ended.sum()),
            completed_service_duration_sum=float(ended_ages.sum()),
            completed_service_duration_max=float(ended_ages.max()) if len(ended_ages) else 0.,
            service_age_mean=float(self.service_age[served].mean()) if served.any() else 0.,
            score_mean=float(scores[before].mean()) if before.any() else 0.,
            score_std=float(scores[before].std()) if before.any() else 0.)
        self._install(*links, self._graph(links, c))
        self.proposals, self.last_scores = proposals, scores
        return self.obs.copy(), self.state.copy(), self.info["reward"], self.t == self.cfg.horizon, False, self.info.copy()

    def step(self, action):
        prepared = self.prepare(action)
        return self.finish(prepared, physical_links(self.cfg, self.uav_xy, self.user_xy))

    def snapshot(self):
        return dict(super().snapshot(), service_version="topk-score-v1", association=self.C.copy(),
            service_age=self.service_age.copy(), last_owner=self.last_owner.copy(),
            proposals=self.proposals.copy(), last_scores=self.last_scores.copy(),
            service_diagnostics=copy.deepcopy(self.service_diagnostics))

    def restore(self, data):
        if data.get("service_version") != "topk-score-v1":
            raise ValueError("Service snapshot requires actual association and protocol version")
        self.C = np.asarray(data["association"]).copy()
        self.service_age = data["service_age"].copy()
        self.last_owner = data["last_owner"].copy()
        self.proposals = data.get("proposals", np.zeros_like(self.C, bool)).copy()
        self.last_scores = data.get("last_scores", np.zeros_like(self.C, dtype=np.float64)).copy()
        self.service_diagnostics = copy.deepcopy(data["service_diagnostics"])
        self._initializing = False
        super().restore(data)


def step_batch(envs, actions):
    prepared = [env.prepare(a) for env, a in zip(envs, actions)]
    links = physical_links(envs[0].cfg, np.stack([e.uav_xy for e in envs]),
                            np.stack([e.user_xy for e in envs]))
    return [e.finish(p, tuple(x[i] for x in links)) for i, (e, p) in enumerate(zip(envs, prepared))]


class ServiceVector:
    """Batch physics by default; spawn workers use the exact scalar semantics."""
    def __init__(self, cfg, count, backend="batch"):
        self.cfg, self.count, self.backend = cfg, count, backend
        self.envs, self.pipes, self.processes = [], [], []
        if backend == "process":
            import multiprocessing as mp
            for _ in range(count):
                parent, child = mp.get_context("spawn").Pipe()
                process = mp.get_context("spawn").Process(target=_worker, args=(child, cfg), daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
        else:
            self.envs = [ServiceEnvironment(cfg) for _ in range(count)]

    def _rpc(self, command, values):
        for pipe, value in zip(self.pipes, values):
            pipe.send((command, value))
        result = []
        for pipe in self.pipes:
            ok, value = pipe.recv()
            if not ok:
                raise RuntimeError(value)
            result.append(value)
        return result

    def reset(self, seeds):
        if len(seeds) != self.count:
            raise ValueError("One seed per environment required")
        rows = self._rpc("reset", seeds) if self.pipes else [e.reset(s) for e, s in zip(self.envs, seeds)]
        return stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), [r[2] for r in rows]

    def reset_at(self, i):
        if self.pipes:
            self.pipes[i].send(("reset", None))
            ok, row = self.pipes[i].recv()
            if not ok:
                raise RuntimeError(row)
            return row
        return self.envs[i].reset()

    def step(self, action):
        actions = action.split(self.cfg.num_uavs)
        if len(actions) != self.count:
            raise ValueError("Wrong service action environment count")
        rows = self._rpc("step", actions) if self.pipes else step_batch(self.envs, actions)
        return (stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
                np.asarray([r[2] for r in rows], np.float32),
                np.asarray([r[3] for r in rows]), np.asarray([r[4] for r in rows]), [r[5] for r in rows])

    def snapshot(self):
        return self._rpc("snapshot", [None] * self.count) if self.pipes else [e.snapshot() for e in self.envs]

    def restore(self, states):
        if len(states) != self.count:
            raise ValueError("Cannot change environment count during resume")
        if self.pipes:
            self._rpc("restore", states)
        else:
            for e, s in zip(self.envs, states):
                e.restore(s)

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
            p.close()
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join()


def _worker(pipe, cfg):
    import numba
    numba.set_num_threads(1)
    env = ServiceEnvironment(cfg)
    try:
        while True:
            name, value = pipe.recv()
            if name == "close":
                break
            result = env.snapshot() if name == "snapshot" else getattr(env, name)(value)
            pipe.send((True, result))
    except BaseException as exc:
        pipe.send((False, repr(exc)))
    finally:
        pipe.close()
