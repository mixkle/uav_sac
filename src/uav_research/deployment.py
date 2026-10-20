"""Actor-only artifacts: no critic and no global UE population dependency."""

import torch
from .models import GaussianActor, DeterministicActor, CategoricalActor
from .actions import action_specification, validate_action_specification
from .config import config_from_dict
from .observations import OBSERVATION_VERSION, as_tensor_observation
from .runtime import atomic_save, load_checkpoint


def export_actor(checkpoint, output):
    data = load_checkpoint(checkpoint)
    e, a = data["config"]["env"], data["config"]["algo"]
    cfg = config_from_dict(data["config"])
    validate_action_specification(data, cfg.env, cfg.algo.name)
    actor = data["learner"]["actor"]
    # Infer the fixed dimension from the saved actor input, rather than population M.
    fixed_dim = actor["net.0.weight"].shape[1] - a["ue_hidden"] - 1
    specification = dict(
        fixed_dim=fixed_dim,
        hidden=a["hidden"],
        ue_hidden=a["ue_hidden"],
        count_scale=e["capacity"],
        log_std_min=a["log_std_min"],
        log_std_max=a["log_std_max"],
        algorithm=a["name"],
    )
    artifact = dict(
        kind="actor",
        action_specification=action_specification(cfg.env, cfg.algo.name),
        observation_version=OBSERVATION_VERSION,
        specification=specification,
        actor=actor,
        scales={
            k: e[k]
            for k in ("width", "height", "max_speed", "dt", "capacity", "sinr_scale_db")
        },
        num_uavs=e["num_uavs"],
        bs_positions=e["bs_positions"],
        env_steps=data["env_steps"],
    )
    from pathlib import Path

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_save(output, artifact)
    return dict(
        output=str(output),
        env_steps=data["env_steps"],
        observation_version=OBSERVATION_VERSION,
    )


class DeployedActor:
    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.artifact = load_checkpoint(path)
        a = self.artifact["specification"]
        self.deterministic_actor = a["algorithm"] == "matd3"
        mode = self.artifact.get("action_specification", {}).get(
            "action_space", "continuous"
        )
        if mode not in ("continuous", "discrete9"):
            raise ValueError("Unknown actor action space")
        if "action_specification" in self.artifact:
            expected = action_specification(
                type("ActorEnv", (), {"action_space": mode})(), a["algorithm"]
            )
            if self.artifact["action_specification"] != expected:
                raise ValueError("Incompatible actor action mapping/distribution")
        if mode == "discrete9":
            expected = action_specification(
                type("DiscreteEnv", (), {"action_space": "discrete9"})(), a["algorithm"]
            )
            if (
                self.artifact["action_specification"] != expected
                or a["algorithm"] != "mappo"
            ):
                raise ValueError("Incompatible discrete actor action specification")
            self.actor = CategoricalActor(
                a["fixed_dim"], a["hidden"], a["ue_hidden"], a["count_scale"]
            )
        elif self.deterministic_actor:
            self.actor = DeterministicActor(
                a["fixed_dim"], a["hidden"], a["ue_hidden"], a["count_scale"]
            )
        else:
            self.actor = GaussianActor(
                a["fixed_dim"],
                a["hidden"],
                a["log_std_min"],
                a["log_std_max"],
                state_dependent_std=a["algorithm"] != "mappo",
                ue_hidden=a["ue_hidden"],
                count_scale=a["count_scale"],
            )
        self.actor.load_state_dict(self.artifact["actor"])
        self.actor.to(self.device).eval()

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        obs = as_tensor_observation(obs, self.device)
        action = (
            self.actor(obs)
            if self.deterministic_actor
            else self.actor.sample(obs, deterministic)[0]
        )
        return action.cpu().numpy()
