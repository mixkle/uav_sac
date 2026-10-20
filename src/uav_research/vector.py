from __future__ import annotations
import multiprocessing as mp
import numpy as np
from .config import EnvConfig
from .actions import decode_actions
from .environment import UAVEnvironment, physical_links
from .kernels import graph_batch
from .observations import stack_observations


def _worker(pipe, cfg):
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:
        pass
    env = UAVEnvironment(cfg)
    try:
        while True:
            command, value = pipe.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset(value)
            elif command == "step":
                result = env.step(value)
            elif command == "snapshot":
                result = env.snapshot()
            elif command == "restore":
                env.restore(value)
                result = None
            else:
                raise ValueError(command)
            pipe.send((True, result))
    except BaseException as error:
        pipe.send((False, repr(error)))
    finally:
        pipe.close()


class VectorEnvironment:
    """Explicit reset: terminal observations are stored before any reset."""

    def __init__(self, cfg: EnvConfig, count: int, backend="batch"):
        self.cfg, self.count, self.backend = cfg, count, backend
        self.processes, self.pipes = [], []
        if backend == "process":
            context = mp.get_context("spawn")
            for _ in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker, args=(child, cfg), daemon=True
                )
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
            self.envs = []
        else:
            self.envs = [UAVEnvironment(cfg) for _ in range(count)]

    def _receive(self, index):
        ok, value = self.pipes[index].recv()
        if not ok:
            raise RuntimeError(f"Environment worker {index}: {value}")
        return value

    def reset(self, seeds):
        if len(seeds) != self.count:
            raise ValueError("One seed per environment required")
        if self.backend == "process":
            for pipe, seed in zip(self.pipes, seeds):
                pipe.send(("reset", int(seed)))
            result = [self._receive(i) for i in range(self.count)]
        else:
            result = [env.reset(int(seed)) for env, seed in zip(self.envs, seeds)]
        return (
            stack_observations([x[0] for x in result]),
            np.stack([x[1] for x in result]),
            [x[2] for x in result],
        )

    def reset_at(self, index):
        if self.backend == "process":
            self.pipes[index].send(("reset", None))
            return self._receive(index)
        return self.envs[index].reset()

    def step(self, actions):
        if self.cfg.action_space == "discrete9":
            decode_actions(actions, (self.count, self.cfg.num_uavs))
        elif np.shape(actions) != (self.count, self.cfg.num_uavs, 2):
            raise ValueError("Expected [E,U,2] vector actions")
        if self.backend == "process":
            for pipe, action in zip(self.pipes, actions):
                pipe.send(("step", action))
            result = [self._receive(i) for i in range(self.count)]
        elif self.cfg.backend == "numba" and self.count > 1:
            for env, action in zip(self.envs, actions):
                env._move(action)
            links = physical_links(
                self.cfg,
                np.stack([e.uav_xy for e in self.envs]),
                np.stack([e.user_xy for e in self.envs]),
            )
            graphs = graph_batch(
                links[0],
                links[1],
                np.where(links[5] >= self.cfg.sensing_threshold_db, links[2], -np.inf),
                self.cfg.sinr_threshold_db,
                self.cfg.capacity,
                self.cfg.num_users,
            )
            result = []
            for i, env in enumerate(self.envs):
                env._install(*(x[i] for x in links), tuple(x[i] for x in graphs))
                result.append(
                    (
                        env.obs.copy(),
                        env.state.copy(),
                        env.info["reward"],
                        env.t == self.cfg.horizon,
                        False,
                        env.info.copy(),
                    )
                )
        else:
            result = [env.step(action) for env, action in zip(self.envs, actions)]
        return (
            stack_observations([x[0] for x in result]),
            np.stack([x[1] for x in result]),
            np.array([x[2] for x in result], np.float32),
            np.array([x[3] for x in result]),
            np.array([x[4] for x in result]),
            [x[5] for x in result],
        )

    def snapshot(self):
        if self.backend == "process":
            for pipe in self.pipes:
                pipe.send(("snapshot", None))
            return [self._receive(i) for i in range(self.count)]
        return [env.snapshot() for env in self.envs]

    def restore(self, snapshots):
        if len(snapshots) != self.count:
            raise ValueError("Environment count changed during resume")
        if self.backend == "process":
            for pipe, snapshot in zip(self.pipes, snapshots):
                pipe.send(("restore", snapshot))
            for i in range(self.count):
                self._receive(i)
        else:
            for env, snapshot in zip(self.envs, snapshots):
                env.restore(snapshot)

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for pipe in self.pipes:
            pipe.close()
        self.pipes, self.processes = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
