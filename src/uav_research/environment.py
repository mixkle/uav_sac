from __future__ import annotations
from .boundary import boundary_distances, directional_boundary_cost, outward_components, reward_contributions

import copy
import numpy as np
from .config import EnvConfig
from .actions import decode_actions
from . import kernels
from .observations import RawObservation, history_specification
from .motion import hop_speed_factor, motion_contract

C_LIGHT = 299792458.0


def reflect(position, bounds):
    """Specular reflection handles arbitrarily large overshoots."""
    phase = np.mod(position, 2 * np.asarray(bounds))
    return np.minimum(phase, 2 * np.asarray(bounds) - phase)


def uniform_disk(rng, shape):
    angle = rng.uniform(0, 2 * np.pi, shape)
    radius = np.sqrt(rng.uniform(0, 1, shape))
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1).astype(
        np.float32
    )


def initialize_users(cfg: EnvConfig, rng):
    """Full/interior uniform or truncated Gaussian; no service screening.

    Centers, sigma and margin use region fractions. The margin applies only at
    reset: subsequent UE motion reflects at the original physical boundaries.
    """
    scale = np.array([cfg.width, cfg.height])
    if cfg.user_initialization == "uniform":
        return rng.uniform([0.0, 0.0], scale, (cfg.num_users, 2))
    if cfg.user_initialization == "uniform_interior":
        lower = cfg.user_initial_margin_fraction * scale
        return rng.uniform(lower, scale - lower, (cfg.num_users, 2))
    if cfg.user_initialization != "gaussian_mixture":
        raise ValueError("Unknown UE initialization")
    centers = np.asarray(cfg.user_cluster_centers, np.float64) * scale
    labels = rng.integers(len(centers), size=cfg.num_users)
    xy = rng.normal(centers[labels], cfg.user_cluster_std_fraction * scale)
    lower = cfg.user_initial_margin_fraction * scale
    upper = scale - lower
    for _ in range(10000):
        invalid = np.any((xy < lower) | (xy > upper), axis=1)
        if not invalid.any():
            return xy
        xy[invalid] = rng.normal(
            centers[labels[invalid]], cfg.user_cluster_std_fraction * scale
        )
    raise RuntimeError(
        "UE Gaussian sampling failed: check initialization margin and dispersion"
    )


def physical_links(cfg: EnvConfig, uav_xy, user_xy):
    """Supports [U,2] or [E,U,2]; all physical calculations are float64."""
    uav_xy, user_xy = np.asarray(uav_xy, np.float64), np.asarray(user_xy, np.float64)
    delta = uav_xy[..., :, None, :] - user_xy[..., None, :, :]
    horizontal = np.linalg.norm(delta, axis=-1)
    distance = np.sqrt(horizontal**2 + cfg.altitude**2)
    elevation = np.degrees(np.arctan2(cfg.altitude, horizontal))
    p_los = 1 / (1 + cfg.los_a * np.exp(-cfg.los_b * (elevation - cfg.los_a)))
    loss = 20 * np.log10(4 * np.pi * cfg.frequency_hz * distance / C_LIGHT)
    loss += p_los * cfg.los_extra_db + (1 - p_los) * cfg.nlos_extra_db
    received = 10 ** ((cfg.subband_power_dbm - loss) / 10)
    # Explicit exclusion avoids catastrophic cancellation when one UAV dominates.
    interference = np.zeros_like(received)
    prefix = np.cumsum(received, axis=-2)
    suffix = np.cumsum(received[..., ::-1, :], axis=-2)[..., ::-1, :]
    interference[..., 1:, :] += prefix[..., :-1, :]
    interference[..., :-1, :] += suffix[..., 1:, :]
    sinr = 10 * np.log10(received / (cfg.noise_mw + interference))
    rate = (
        (cfg.bandwidth_hz / cfg.capacity)
        * np.logaddexp(0, sinr * np.log(10) / 10)
        / np.log(2)
    )
    uu_delta = uav_xy[..., :, None, :] - uav_xy[..., None, :, :]
    adjacency = np.sum(uu_delta**2, axis=-1) <= cfg.uu_range**2
    bs_xy = np.asarray(cfg.bs_positions, np.float64)[:, :2]
    ug_dist2 = np.sum((uav_xy[..., :, None, :] - bs_xy) ** 2, axis=-1) + cfg.altitude**2
    direct = ug_dist2 <= cfg.ug_range**2
    uplink = cfg.beacon_power_dbm - loss - cfg.beacon_noise_dbm
    return adjacency, direct, sinr, rate, loss, uplink


class UAVEnvironment:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        if cfg.backend == "numba" and not kernels.NUMBA_AVAILABLE:
            raise RuntimeError(
                "Numba backend requested: install numba or select backend=numpy"
            )
        self.rng = np.random.default_rng(0)
        self.movement_rng = np.random.default_rng(np.random.SeedSequence([0, 201]))
        self.initial_seed = 0
        self.uav_xy = np.zeros((cfg.num_uavs, 2), np.float64)
        self.user_xy = np.zeros((cfg.num_users, 2), np.float64)
        self.t = 0
        self.last_action_ids = np.full(cfg.num_uavs, 8, np.int64)
        self.last_action = np.zeros((cfg.num_uavs, 2), np.float32)
        self.actual_displacement = np.zeros_like(self.uav_xy)
        self.action_origin_xy = self.uav_xy.copy()
        self.boundary_cost_per_uav = np.zeros(cfg.num_uavs, np.float64)
        self.history = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.movement_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 201])
            )
            self.initial_seed = int(seed)
        e = self.cfg
        self.t = 0
        self.history = None
        self.boundary_cost_per_uav.fill(0)
        self.pre_action_hops = np.full(e.num_uavs, e.num_uavs+1, np.int64)
        self.speed_factor = np.ones(e.num_uavs)
        self.command_displacement = np.zeros((e.num_uavs, 2))
        self.effective_displacement = np.zeros((e.num_uavs, 2))
        self.last_action.fill(0)
        self.last_action_ids.fill(8)
        self.actual_displacement.fill(0)
        self.user_xy = initialize_users(e, self.rng)
        self.initial_user_xy = self.user_xy.copy()
        self.user_speed_mean = 0.0
        if e.uav_initialization == "uniform":
            self.uav_xy = self.rng.uniform([0, 0], [e.width, e.height], (e.num_uavs, 2))
        else:
            radius = e.launch_radius * np.sqrt(self.rng.random(e.num_uavs))
            angle = self.rng.uniform(0, np.pi / 2, e.num_uavs)
            self.uav_xy = np.column_stack(
                [radius * np.cos(angle), radius * np.sin(angle)]
            )
            self.uav_xy[e.num_uavs // 2 :, 1] = (
                e.height - self.uav_xy[e.num_uavs // 2 :, 1]
            )
        self.action_origin_xy = self.uav_xy.copy()
        self.initial_uav_xy = self.uav_xy.copy()
        self.ever_discovered = np.zeros(e.num_users, bool)
        self.first_service_step = -1
        self.zero_coverage_steps = 0
        self._refresh()
        return self.obs.copy(), self.state.copy(), self.info.copy()

    def set_positions(self, uav_xy, user_xy, t=0):
        """Explicit deterministic scenes for model tests and diagnostics; bypasses reset conditioning."""
        e = self.cfg
        u, m = np.asarray(uav_xy, np.float64), np.asarray(user_xy, np.float64)
        if u.shape != (e.num_uavs, 2) or m.shape != (e.num_users, 2):
            raise ValueError("Position shape mismatch")
        if not np.isfinite(u).all() or not np.isfinite(m).all():
            raise ValueError("Nonfinite positions")
        bounds = np.array([e.width, e.height])
        if (u < 0).any() or (u > bounds).any() or (m < 0).any() or (m > bounds).any():
            raise ValueError("Positions outside region")
        if not 0 <= t <= e.horizon:
            raise ValueError("Invalid task time")
        self.uav_xy, self.user_xy, self.t = u.copy(), m.copy(), int(t)
        self.history = None
        self.initial_user_xy = self.user_xy.copy()
        self.user_speed_mean = 0.0
        self.action_origin_xy = self.uav_xy.copy()
        self.initial_uav_xy = self.uav_xy.copy()
        self.ever_discovered = np.zeros(e.num_users, bool)
        self.first_service_step = -1
        self.zero_coverage_steps = 0
        self.pre_action_hops = np.full(e.num_uavs, e.num_uavs+1, np.int64)
        self.speed_factor = np.ones(e.num_uavs)
        self.command_displacement = np.zeros((e.num_uavs, 2))
        self.effective_displacement = np.zeros((e.num_uavs, 2))
        self.last_action.fill(0)
        self.actual_displacement.fill(0)
        self.boundary_cost_per_uav.fill(0)
        self._refresh()
        return self.obs.copy(), self.state.copy(), self.info.copy()

    def _move(self, action):
        e = self.cfg
        if e.action_space == "discrete9":
            command = decode_actions(action, (e.num_uavs,))
            self.last_action_ids = np.asarray(action).astype(np.int64, copy=True)
            action = command
        action = np.asarray(action, np.float32)
        if action.shape != (e.num_uavs, 2) or not np.isfinite(action).all():
            raise ValueError("Expected finite [U,2] actions")
        if np.any(np.linalg.norm(action, axis=-1) > 1 + 1e-6):
            raise ValueError("Actions must lie in the normalized unit disk")
        if self.t >= e.horizon:
            raise RuntimeError("Task ended: reset before stepping")
        if e.observation_history == "one_step":
            self.history = np.column_stack([
                action, self.B.min(axis=1) / (e.num_uavs + 1),
                self.sensed.sum(axis=1) / e.capacity,
            ]).astype(np.float32)
        previous = self.uav_xy.copy()
        self.action_origin_xy = previous.copy()
        self.boundary_cost_per_uav = directional_boundary_cost(
            previous, action, e.width, e.height, e.boundary_penalty_margin_m)
        self.pre_action_hops = self.B.min(1).copy()
        ms = e.motion_scaling
        self.speed_factor = (hop_speed_factor(self.pre_action_hops, e.num_uavs, ms["min_ratio"], ms["hop_growth"])
                             if ms["enabled"] else np.ones(e.num_uavs))
        self.command_displacement = action * e.max_speed * e.dt
        self.effective_displacement = self.command_displacement.astype(np.float64)*self.speed_factor[:, None]
        # Preserve historical floating arithmetic exactly when disabled.
        proposed = previous + (self.effective_displacement if ms["enabled"] else self.command_displacement)
        self.uav_xy = (
            reflect(proposed, [e.width, e.height])
            if e.uav_boundary == "reflect"
            else np.clip(proposed, [0, 0], [e.width, e.height])
        )
        self.actual_displacement = self.uav_xy - previous
        if e.observation_history == "one_step" and e.history_motion_source == "post_reflection_displacement":
            self.history[:, :2] = self.actual_displacement/(e.max_speed*e.dt)
        self.last_action = action.copy()
        v = np.clip(
            self.movement_rng.normal(0, e.user_speed_std, e.num_users),
            -e.user_speed_clip,
            e.user_speed_clip,
        )
        direction = self.movement_rng.uniform(0, 2 * np.pi, e.num_users)
        displacement = (
            np.column_stack([v * np.cos(direction), v * np.sin(direction)]) * e.dt
        )
        next_users = reflect(self.user_xy + displacement, [e.width, e.height])
        self.user_speed_mean = float(np.linalg.norm(next_users - self.user_xy, axis=1).mean() / e.dt)
        self.user_xy = next_users
        self.t += 1

    def step(self, action):
        self._move(action)
        self._refresh()
        return (
            self.obs.copy(),
            self.state.copy(),
            self.info["reward"],
            self.t == self.cfg.horizon,
            False,
            self.info.copy(),
        )

    def _refresh(self):
        e = self.cfg
        a, direct, y, rates, loss, uplink = physical_links(e, self.uav_xy, self.user_xy)
        candidates_y = np.where(uplink >= e.sensing_threshold_db, y, -np.inf)
        if e.backend == "numba":
            h = kernels.shortest_hops(a, direct, np.ones(e.num_uavs, bool))
            c, n, degree = kernels.associate(
                candidates_y, h, e.sinr_threshold_db, e.capacity
            )
            retained, switched, post = kernels.failure_coverage(
                a, direct, n, h, e.num_users
            )
            labels = kernels.component_labels(a)
            graph = h, c, n, degree, retained, switched, post, labels
        else:
            graph = kernels.reference_graph(
                a, direct, candidates_y, e.sinr_threshold_db, e.capacity, e.num_users
            )
        self._install(a, direct, y, rates, loss, uplink, graph)

    def _install(self, a, direct, y, rates, loss, uplink, graph):
        e = self.cfg
        self.A, self.direct, self.Y, self.rates, self.loss = a, direct, y, rates, loss
        (
            self.B,
            self.C,
            self.counts,
            self.candidate_counts,
            self.retained,
            self.switched,
            self.post_hops,
            self.labels,
        ) = graph
        best_hop = self.B.min(axis=1)
        coverage = float(self.counts.sum() / e.num_users)
        throughput = float(np.sum(self.C * self.rates))
        tau = throughput / (e.num_users * e.rate_reference_bps)
        robust = float(self.retained.mean())
        disconnected_uavs = int((best_hop > e.num_uavs).sum())
        disconnection_ratio = disconnected_uavs / e.num_uavs
        disconnection_penalty = e.disconnection_weight * disconnection_ratio
        reached = self.B <= e.num_uavs
        component_services = np.bincount(self.labels, weights=self.counts)
        xy_scale = np.array([e.width, e.height])
        self.uplink = uplink
        self.sensed = uplink >= e.sensing_threshold_db
        builder = (
            kernels.build_observation
            if e.backend == "numba"
            else kernels.build_observation.py_func
        )
        self.obs = RawObservation(
            *builder(
                self.uav_xy,
                self.user_xy,
                self.A,
                self.B,
                self.counts,
                self.C,
                uplink,
                self.sensed,
                np.asarray(e.bs_positions)[:, :2],
                e.width,
                e.height,
                e.capacity,
                e.neighbor_slots,
                e.sinr_scale_db,
                e.observation_coordinates == "relative",
            )
        )
        if e.observation_history == "one_step":
            if self.history is None:
                self.history = np.column_stack([
                    np.zeros((e.num_uavs, 2)), best_hop / (e.num_uavs + 1),
                    self.sensed.sum(axis=1) / e.capacity,
                ]).astype(np.float32)
            self.obs.fixed = np.concatenate([self.obs.fixed, self.history], axis=-1)
        seen = self.sensed.any(axis=0)
        discovery_fraction = float(seen.mean())
        primary_reward = discovery_fraction if e.reward_primary == "sensing" else coverage
        self.ever_discovered |= seen
        distances = np.linalg.norm(self.uav_xy[:, None] - self.uav_xy[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        if coverage > 0 and self.first_service_step < 0:
            self.first_service_step = self.t
        if self.t > 0 and coverage == 0:
            self.zero_coverage_steps += 1
        down = self.sensed & (self.Y >= e.sinr_threshold_db)
        accessible = down & (best_hop[:, None] <= e.num_uavs)
        xyz_u = np.column_stack([self.uav_xy / xy_scale, np.ones(e.num_uavs)])
        xyz_m = np.column_stack([self.user_xy / xy_scale, np.zeros(e.num_users)])
        xyz_g = np.asarray(e.bs_positions) / np.array([e.width, e.height, e.altitude])
        self.state = np.concatenate(
            [
                xyz_u.ravel(),
                xyz_g.ravel(),
                xyz_m.ravel(),
                self.A.ravel(),
                (self.B / (e.num_uavs + 1)).ravel(),
                self.C.ravel(),
                (self.Y / e.sinr_scale_db).ravel(),
                [self.t / e.horizon],
            ]
        ).astype(np.float32)
        sinr_percentiles = np.percentile(self.Y, [10, 50, 90])
        task_reward = primary_reward + e.throughput_weight * tau + e.robustness_weight * robust - disconnection_penalty
        boundary_cost = float(self.boundary_cost_per_uav.mean())
        boundary_penalty = e.boundary_penalty_weight * boundary_cost
        distance = boundary_distances(self.uav_xy, e.width, e.height).min(-1)
        pre_distance = boundary_distances(self.action_origin_xy, e.width, e.height)
        in_buffer = pre_distance < e.boundary_penalty_margin_m
        edge = boundary_distances(self.user_xy, e.width, e.height).min(-1) < 150.
        served = self.C.any(axis=0)
        info = dict(
            task_reward=task_reward, boundary_cost=boundary_cost, boundary_penalty=boundary_penalty,
            boundary_distance_mean=float(distance.mean()),
            boundary_within_50_fraction=float((distance < 50).mean()),
            boundary_within_100_fraction=float((distance < 100).mean()),
            boundary_within_150_fraction=float((distance < 150).mean()),
            boundary_buffer_fraction=float(in_buffer.any(-1).mean()),
            boundary_outward_fraction=float((in_buffer & (outward_components(self.last_action) > 1e-8)).any(-1).mean()),
            edge_users=int(edge.sum()), edge_served_users=int((edge & served).sum()),
            interior_users=int((~edge).sum()), interior_served_users=int((~edge & served).sum()),
            discovered_users=int(seen.sum()),
            discovery_fraction=discovery_fraction,
            cumulative_discovery_fraction=float(self.ever_discovered.mean()),
            discovered_unserved_users=int((seen & ~self.C.any(axis=0)).sum()),
            sensing_rejected_users=int((~seen).sum()),
            downlink_rejected_users=int((seen & ~down.any(axis=0)).sum()),
            backhaul_rejected_users=int(
                (down.any(axis=0) & ~accessible.any(axis=0)).sum()
            ),
            capacity_rejected_users=int(
                (accessible.any(axis=0) & ~self.C.any(axis=0)).sum()
            ),
            first_service_step=int(self.first_service_step),
            zero_coverage_steps=int(self.zero_coverage_steps),
            expansion_distance_mean=float(
                np.linalg.norm(self.uav_xy - self.initial_uav_xy, axis=1).mean()
            ),
            actual_speed_mean=float(
                np.linalg.norm(self.actual_displacement, axis=1).mean() / e.dt
            ),
            user_speed_mean=self.user_speed_mean,
            user_displacement_mean=float(np.linalg.norm(self.user_xy - self.initial_user_xy, axis=1).mean()),
            uav_nearest_neighbor_mean=float(nearest.mean()) if e.num_uavs > 1 else 0.0,
            uav_close_neighbor_fraction=float((nearest < 100.0).mean()),
            observed_count_mean=float(self.sensed.sum(axis=1).mean()),
            observed_count_max=int(self.sensed.sum(axis=1).max()),
            reward_primary=primary_reward,
            reward=task_reward - boundary_penalty,
            disconnection_ratio=disconnection_ratio,
            disconnection_penalty=disconnection_penalty,
            coverage=coverage,
            agent_coverage=(self.counts / e.num_users).tolist(),
            throughput_bps=throughput,
            tau=tau,
            robustness=robust,
            retention_ratio=robust / coverage if coverage else 0.0,
            worst_fault_coverage=float(self.retained.min()),
            disconnected_uavs=disconnected_uavs,
            mean_hops=float(best_hop[best_hop <= e.num_uavs].mean())
            if (best_hop <= e.num_uavs).any()
            else 0.0,
            capacity_utilization=float(self.counts.sum() / (e.num_uavs * e.capacity)),
            component_count=int(self.labels.max() + 1),
            component_services=component_services.tolist(),
            gateway_switch_coverage=float(self.switched.mean()),
            reachable_all_gateways=int(reached.all(axis=1).sum()),
            reachable_none=int((~reached.any(axis=1)).sum()),
            candidate_count_mean=float(self.candidate_counts.mean()),
            candidate_count_max=int(self.candidate_counts.max()),
            sinr_db_p10=float(sinr_percentiles[0]),
            sinr_db_p50=float(sinr_percentiles[1]),
            sinr_db_p90=float(sinr_percentiles[2]),
            boundary_reflection_fraction=float(e.uav_boundary == "reflect") * float(
                np.any(np.abs(self.actual_displacement - self.effective_displacement) > 1e-5, axis=1).mean()
            ),
            speed_factor_mean=float(self.speed_factor.mean()),
            effective_action_norm_mean=float((np.linalg.norm(self.last_action, axis=1)*self.speed_factor).mean()),
            pre_action_connected_fraction=float((self.pre_action_hops <= e.num_uavs).mean()),
            newly_disconnected_fraction=float(((self.pre_action_hops <= e.num_uavs) & (self.B.min(1) > e.num_uavs)).mean()) if self.t else 0.,
            action_norm_mean=float(np.linalg.norm(self.last_action, axis=1).mean()),
            boundary_clip_fraction=float(e.uav_boundary == "clip") * float(
                np.mean(
                    np.any(
                        np.abs(
                            self.actual_displacement
                            - self.effective_displacement
                        )
                        > 1e-5,
                        axis=1,
                    )
                )
            ),
        )
        for g in range(e.num_bs):
            info[f"bs_{g}_direct_uavs"] = int(self.direct[:, g].sum())
            info[f"bs_{g}_only_reachable"] = int(
                (reached[:, g] & (reached.sum(axis=1) == 1)).sum()
            )
        self.info = info
        if e.action_space == "discrete9":
            self.info.update(
                {
                    f"action_{i}_frequency": float((self.last_action_ids == i).mean())
                    for i in range(9)
                }
            )
            self.info["hover_fraction"] = float((self.last_action_ids == 8).mean())

    def snapshot(self):
        return dict(motion_contract=motion_contract(self.cfg),
            pre_action_hops=self.pre_action_hops.copy(), speed_factor=self.speed_factor.copy(),
            command_displacement=self.command_displacement.copy(), effective_displacement=self.effective_displacement.copy(),
            **(dict(observation_history="one_step", history_specification=history_specification(self.cfg), history=self.history.copy())
               if self.cfg.observation_history == "one_step" else {}),
            uav_xy=self.uav_xy.copy(),
            user_xy=self.user_xy.copy(),
            initial_user_xy=self.initial_user_xy.copy(),
            user_speed_mean=self.user_speed_mean,
            t=self.t,
            initial_uav_xy=self.initial_uav_xy.copy(),
            ever_discovered=self.ever_discovered.copy(),
            first_service_step=self.first_service_step,
            zero_coverage_steps=self.zero_coverage_steps,
            action=self.last_action.copy(),
            **(
                {"action_ids": self.last_action_ids.copy()}
                if self.cfg.action_space == "discrete9"
                else {}
            ),
            actual_displacement=self.actual_displacement.copy(),
            action_origin_xy=self.action_origin_xy.copy(),
            boundary_cost_per_uav=self.boundary_cost_per_uav.copy(),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            movement_rng=copy.deepcopy(self.movement_rng.bit_generator.state),
            initial_seed=self.initial_seed,
        )

    def restore(self, data):
        expected = motion_contract(self.cfg)
        old_default = dict(motion_scaling=dict(enabled=False, min_ratio=.25, hop_growth=.4), history_motion_source="command")
        if data.get("motion_contract", old_default) != expected:
            raise ValueError("Snapshot motion/history contract mismatch")
        for name, default in (("pre_action_hops", np.full(self.cfg.num_uavs,self.cfg.num_uavs+1)),
                ("speed_factor", np.ones(self.cfg.num_uavs)),
                ("command_displacement", data["action"]*self.cfg.max_speed*self.cfg.dt),
                ("effective_displacement", data["action"]*self.cfg.max_speed*self.cfg.dt)):
            if name not in data and expected != old_default:
                raise ValueError("Snapshot requires complete executed motion metadata")
            setattr(self, name, np.asarray(data.get(name, default)).copy())
        if data.get("observation_history", "none") != self.cfg.observation_history:
            raise ValueError("Snapshot observation_history does not match environment")
        if self.cfg.observation_history == "one_step":
            history = np.asarray(data.get("history"), dtype=np.float32)
            if data.get("history_specification") != history_specification(self.cfg):
                raise ValueError("Snapshot history specification does not match environment")
            if history.shape != (self.cfg.num_uavs, 4) or not np.isfinite(history).all():
                raise ValueError("Snapshot requires complete finite one_step history")
            self.history = history.copy()
        self.uav_xy, self.user_xy, self.t = (
            data["uav_xy"].copy(),
            data["user_xy"].copy(),
            data["t"],
        )
        self.last_action = data["action"].copy()
        self.last_action_ids = data.get(
            "action_ids", np.full(self.cfg.num_uavs, 8, np.int64)
        ).copy()
        self.actual_displacement = data["actual_displacement"].copy()
        self.action_origin_xy = data.get("action_origin_xy", self.uav_xy).copy()
        self.boundary_cost_per_uav = data.get("boundary_cost_per_uav", np.zeros(self.cfg.num_uavs)).copy()
        # Old snapshots did not track UE displacement; start that diagnostic here.
        self.initial_user_xy = data.get("initial_user_xy", self.user_xy).copy()
        self.user_speed_mean = data.get("user_speed_mean", 0.0)
        self.rng.bit_generator.state = copy.deepcopy(data["rng"])
        self.movement_rng.bit_generator.state = copy.deepcopy(data["movement_rng"])
        self.initial_seed = data["initial_seed"]
        self.initial_uav_xy = data["initial_uav_xy"].copy()
        self.ever_discovered = data["ever_discovered"].copy()
        self.first_service_step = data["first_service_step"]
        self.zero_coverage_steps = data["zero_coverage_steps"]
        zero_steps = self.zero_coverage_steps
        self._refresh()
        self.zero_coverage_steps = zero_steps
        self.info["zero_coverage_steps"] = zero_steps

    def trace_frame(self):
        return dict(pre_action_hops=self.pre_action_hops.copy(), speed_factor=self.speed_factor.copy(),
            command_displacement=self.command_displacement.copy(), effective_displacement=self.effective_displacement.copy(),
            **(dict(observation_history=self.history.copy())
               if self.cfg.observation_history == "one_step" else {}),
            uav_xy=self.uav_xy.copy(),
            user_xy=self.user_xy.copy(),
            action=self.last_action.copy(),
            **(
                {"action_ids": self.last_action_ids.copy()}
                if self.cfg.action_space == "discrete9"
                else {}
            ),
            actual_displacement=self.actual_displacement.copy(),
            action_origin_xy=self.action_origin_xy.copy(),
            boundary_cost_per_uav=self.boundary_cost_per_uav.copy(),
            A=self.A.copy(),
            B=self.B.copy(),
            C=self.C.copy(),
            Y=self.Y.copy(),
            uplink=self.uplink.copy(),
            sensed=self.sensed.copy(),
            coverage=self.info["coverage"],
            agent_coverage=np.asarray(self.info["agent_coverage"], dtype=np.float64),
            discovery_fraction=self.info["discovery_fraction"],
            cumulative_discovery_fraction=self.info["cumulative_discovery_fraction"],
            reward_primary=self.info["reward_primary"],
            user_speed_mean=self.info["user_speed_mean"],
            user_displacement_mean=self.info["user_displacement_mean"],
            uav_nearest_neighbor_mean=self.info["uav_nearest_neighbor_mean"],
            uav_close_neighbor_fraction=self.info["uav_close_neighbor_fraction"],
            reward_terms=np.array(
                [self.info["coverage"], self.info["tau"], self.info["robustness"], self.info["disconnection_ratio"]]
            ),
            reward=self.info["reward"],
            reward_contributions=reward_contributions(self.info, self.cfg),
            task_reward=self.info['task_reward'],
            boundary_cost=self.info['boundary_cost'],
            boundary_penalty=self.info['boundary_penalty'],
            disconnection_ratio=self.info["disconnection_ratio"],
            disconnection_penalty=self.info["disconnection_penalty"],
            fault_coverage=self.retained.copy(),
            fault_hops=self.post_hops.copy(),
        )
