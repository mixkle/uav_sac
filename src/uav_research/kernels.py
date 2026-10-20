"""Graph/association kernels. BS nodes are destinations, never transit edges."""

from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        return lambda f: f


@njit(cache=True)
def shortest_hops(adjacency, direct, active):
    u, g = direct.shape
    hops = np.full((u, g), u + 1, dtype=np.int64)
    for gateway in range(g):
        queue = np.empty(u, dtype=np.int64)
        head, tail = 0, 0
        for i in range(u):
            if active[i] and direct[i, gateway]:
                hops[i, gateway] = 1
                queue[tail] = i
                tail += 1
        while head < tail:
            i = queue[head]
            head += 1
            for j in range(u):
                if active[j] and adjacency[i, j] and hops[j, gateway] == u + 1:
                    hops[j, gateway] = hops[i, gateway] + 1
                    queue[tail] = j
                    tail += 1
    return hops


@njit(cache=True)
def component_labels(adjacency):
    u = adjacency.shape[0]
    labels = np.full(u, -1, dtype=np.int64)
    count = 0
    for root in range(u):
        if labels[root] >= 0:
            continue
        queue = np.empty(u, dtype=np.int64)
        queue[0] = root
        labels[root] = count
        head, tail = 0, 1
        while head < tail:
            i = queue[head]
            head += 1
            for j in range(u):
                if adjacency[i, j] and labels[j] < 0:
                    labels[j] = count
                    queue[tail] = j
                    tail += 1
        count += 1
    return labels


@njit(cache=True)
def associate(sinr, hops, threshold, capacity):
    u, m = sinr.shape
    best_hop = min_hops(hops)
    candidates = np.zeros(m, dtype=np.int64)
    for j in range(m):
        for i in range(u):
            if best_hop[i] <= u and sinr[i, j] >= threshold:
                candidates[j] += 1
    # Stable counting sort by number of candidates, preserving user-ID ties.
    order = np.empty(m, dtype=np.int64)
    k = 0
    for degree in range(u + 1):
        for j in range(m):
            if candidates[j] == degree:
                order[k] = j
                k += 1
    c = np.zeros((u, m), dtype=np.uint8)
    counts = np.zeros(u, dtype=np.int64)
    for k in range(m):
        j = order[k]
        best = -1
        for i in range(u):
            if best_hop[i] > u or counts[i] >= capacity or sinr[i, j] < threshold:
                continue
            if (
                best < 0
                or sinr[i, j] > sinr[best, j]
                or (sinr[i, j] == sinr[best, j] and best_hop[i] < best_hop[best])
            ):
                best = i
        if best >= 0:
            c[best, j] = 1
            counts[best] += 1
    return c, counts, candidates


@njit(cache=True)
def min_hops(hops):
    out = np.empty(hops.shape[0], dtype=np.int64)
    for i in range(hops.shape[0]):
        value = hops[i, 0]
        for g in range(1, hops.shape[1]):
            value = min(value, hops[i, g])
        out[i] = value
    return out


@njit(cache=True)
def failure_coverage(adjacency, direct, counts, nominal_hops, num_users):
    u, g = direct.shape
    retained = np.zeros(u, dtype=np.float64)
    switched = np.zeros(u, dtype=np.float64)
    post_hops = np.empty((u, u, g), dtype=np.int64)
    for failed in range(u):
        active = np.ones(u, dtype=np.bool_)
        active[failed] = False
        post = shortest_hops(adjacency, direct, active)
        post_hops[failed] = post
        for i in range(u):
            if not active[i]:
                continue
            preferred = 0
            reachable = False
            for gateway in range(g):
                if nominal_hops[i, gateway] < nominal_hops[i, preferred]:
                    preferred = gateway
                if post[i, gateway] <= u:
                    reachable = True
            if reachable:
                retained[failed] += counts[i] / num_users
                if post[i, preferred] > u:
                    switched[failed] += counts[i] / num_users
    return retained, switched, post_hops


@njit(cache=True, parallel=True)
def graph_batch(adjacencies, directs, sinrs, threshold, capacity, num_users):
    b, u, g = directs.shape
    m = sinrs.shape[2]
    hops = np.empty((b, u, g), dtype=np.int64)
    associations = np.empty((b, u, m), dtype=np.uint8)
    counts = np.empty((b, u), dtype=np.int64)
    candidates = np.empty((b, m), dtype=np.int64)
    retained = np.empty((b, u), dtype=np.float64)
    switched = np.empty((b, u), dtype=np.float64)
    post = np.empty((b, u, u, g), dtype=np.int64)
    labels = np.empty((b, u), dtype=np.int64)
    for i in prange(b):
        h = shortest_hops(adjacencies[i], directs[i], np.ones(u, dtype=np.bool_))
        c, n, degree = associate(sinrs[i], h, threshold, capacity)
        f, s, p = failure_coverage(adjacencies[i], directs[i], n, h, num_users)
        hops[i], associations[i], counts[i], candidates[i] = h, c, n, degree
        retained[i], switched[i], post[i] = f, s, p
        labels[i] = component_labels(adjacencies[i])
    return hops, associations, counts, candidates, retained, switched, post, labels


@njit(cache=True)
def build_observation(
    uav_xy,
    user_xy,
    adjacency,
    hops,
    counts,
    association,
    uplink,
    sensed,
    bs_xy,
    width,
    height,
    capacity,
    neighbor_slots,
    sinr_scale,
):
    u, m = association.shape
    self_dim, neighbor_dim = 5 + u, 6 + u
    bs_dim = 2 * len(bs_xy)
    fixed = np.zeros((u, self_dim + bs_dim + neighbor_slots * neighbor_dim), np.float32)
    offsets = np.zeros(u + 1, np.int64)
    for i in range(u):
        offsets[i + 1] = offsets[i] + sensed[i].sum()
        fixed[i, 0], fixed[i, 1], fixed[i, 2] = (
            uav_xy[i, 0] / width,
            uav_xy[i, 1] / height,
            1,
        )
        fixed[i, 3], fixed[i, 4] = hops[i].min() / (u + 1), counts[i] / capacity
        for j in range(u):
            fixed[i, 5 + j] = adjacency[i, j]
        for g in range(len(bs_xy)):
            fixed[i, self_dim + 2 * g] = bs_xy[g, 0] / width
            fixed[i, self_dim + 2 * g + 1] = bs_xy[g, 1] / height
    users = np.empty((offsets[-1], 4), np.float32)
    for i in range(u):
        used = np.zeros(u, np.bool_)
        used[i] = True
        for slot in range(neighbor_slots):
            best, distance = -1, np.inf
            for j in range(u):
                if not used[j] and adjacency[i, j]:
                    d = ((uav_xy[j] - uav_xy[i]) ** 2).sum()
                    if d < distance:
                        best, distance = j, d
            if best < 0:
                break
            used[best] = True
            start = self_dim + bs_dim + slot * neighbor_dim
            fixed[i, start] = (best + 1) / u
            for k in range(self_dim):
                fixed[i, start + 1 + k] = fixed[best, k]
        row = offsets[i]
        for j in range(m):
            if sensed[i, j]:
                users[row, 0] = (user_xy[j, 0] - uav_xy[i, 0]) / width
                users[row, 1] = (user_xy[j, 1] - uav_xy[i, 1]) / height
                users[row, 2] = uplink[i, j] / sinr_scale
                users[row, 3] = association[i, j]
                row += 1
    return fixed, users, offsets


def reference_graph(adjacency, direct, sinr, threshold, capacity, num_users):
    """Uncompiled reference path; route failures have their own Python BFS call."""

    def fn(f):
        return getattr(f, "py_func", f)

    u = adjacency.shape[0]
    h = fn(shortest_hops)(adjacency, direct, np.ones(u, bool))
    c, n, degree = fn(associate)(sinr, h, threshold, capacity)
    post = np.stack(
        [fn(shortest_hops)(adjacency, direct, np.arange(u) != f) for f in range(u)]
    )
    reachable = (post <= u).any(axis=2)
    retained = (reachable * n[None]).sum(axis=1) / num_users
    preferred = h.argmin(axis=1)
    switched = (reachable * (post[:, np.arange(u), preferred] > u) * n[None]).sum(
        axis=1
    ) / num_users
    return h, c, n, degree, retained, switched, post, fn(component_labels)(adjacency)
