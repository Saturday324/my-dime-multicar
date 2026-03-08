import importlib
from typing import Any, Dict, Optional, Type

from common.env_factory import resolve_env_kwargs, try_import_env_registrations


_MA_ENV_SHORTCUTS = {
    "metadrive/MultiAgentMetaDrive": "metadrive.envs.marl_envs.multi_agent_metadrive:MultiAgentMetaDrive",
    "metadrive/MultiAgentRoundaboutEnv": "metadrive.envs.marl_envs.marl_inout_roundabout:MultiAgentRoundaboutEnv",
    "metadrive/MultiAgentIntersectionEnv": "metadrive.envs.marl_envs.marl_intersection:MultiAgentIntersectionEnv",
    "metadrive/MultiAgentBottleneckEnv": "metadrive.envs.marl_envs.marl_bottleneck:MultiAgentBottleneckEnv",
    "metadrive/MultiAgentParkingLotEnv": "metadrive.envs.marl_envs.marl_parking_lot:MultiAgentParkingLotEnv",
    "metadrive/MultiAgentTollgateEnv": "metadrive.envs.marl_envs.marl_tollgate:MultiAgentTollgateEnv",
    "metadrive/MultiAgentBidirectionEnv": "metadrive.envs.marl_envs.marl_bidirection:MultiAgentBidirectionEnv",
    "metadrive/MultiAgentRacingEnv": "metadrive.envs.marl_envs.marl_racing_env:MultiAgentRacingEnv",
    "metadrive/MultiAgentTinyInter": "metadrive.envs.marl_envs.tinyinter:MultiAgentTinyInter",
}


def _resolve_ma_class_spec(env_name: str) -> str:
    if env_name in _MA_ENV_SHORTCUTS:
        return _MA_ENV_SHORTCUTS[env_name]
    if ":" in env_name:
        return env_name
    if env_name.startswith("metadrive.") and env_name.count(".") >= 2:
        return env_name
    raise ValueError(
        f"Unsupported multi-agent env_name: {env_name}. "
        "Use one of metadrive/MultiAgent* shortcuts or a full class spec."
    )


def _resolve_class_from_spec(class_spec: str) -> Type[Any]:
    if ":" in class_spec:
        module_name, class_name = class_spec.split(":", 1)
    else:
        module_name, class_name = class_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)
    return env_class


def create_multiagent_env(
    env_name: str,
    raw_env_kwargs: Optional[Any] = None,
    default_start_seed: Optional[int] = None,
) -> Any:
    try_import_env_registrations()

    env_kwargs: Dict[str, Any] = resolve_env_kwargs(raw_env_kwargs)
    env_kwargs = dict(env_kwargs)
    env_kwargs.setdefault("use_render", False)
    env_kwargs.setdefault("allow_respawn", False)
    env_kwargs.setdefault("num_scenarios", 1)
    if default_start_seed is not None:
        env_kwargs.setdefault("start_seed", int(default_start_seed))

    class_spec = _resolve_ma_class_spec(env_name)
    env_class = _resolve_class_from_spec(class_spec)
    try:
        return env_class(config=env_kwargs)
    except TypeError:
        try:
            return env_class(env_kwargs)
        except TypeError:
            return env_class(**env_kwargs)
