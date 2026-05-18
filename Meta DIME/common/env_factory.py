import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import gymnasium as gym


_METADRIVE_SHORTCUTS = {
    "metadrive": "metadrive.envs:MetaDriveEnv",
    "metadrive/MetaDriveEnv": "metadrive.envs:MetaDriveEnv",
    "metadrive/SafeMetaDriveEnv": "metadrive.envs:SafeMetaDriveEnv",
}


def resolve_env_kwargs(raw_env_kwargs: Any) -> Dict[str, Any]:
    if raw_env_kwargs is None:
        return {}
    try:
        import omegaconf

        if isinstance(raw_env_kwargs, omegaconf.DictConfig):
            return dict(omegaconf.OmegaConf.to_container(raw_env_kwargs, resolve=True))
    except Exception:
        pass
    return dict(raw_env_kwargs)


def _import_optional_package(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def _metadrive_envs_available() -> bool:
    try:
        importlib.import_module("metadrive.envs")
        return True
    except ImportError:
        return False


def _try_import_metadrive() -> bool:
    if _import_optional_package("metadrive") and _metadrive_envs_available():
        return True

    local_repo = Path(__file__).resolve().parents[1] / "metadrive"
    if local_repo.is_dir():
        # Remove a namespace-only metadrive module loaded from cwd first.
        sys.modules.pop("metadrive", None)
        sys.modules.pop("metadrive.envs", None)
        local_repo_str = str(local_repo)
        if local_repo_str not in sys.path:
            sys.path.insert(0, local_repo_str)
        if _import_optional_package("metadrive") and _metadrive_envs_available():
            print(f"Loaded metadrive from local clone: {local_repo}")
            return True
    return False


def try_import_env_registrations() -> None:
    if not _import_optional_package("myosuite"):
        print("myosuite not installed, skipping")
    if not _try_import_metadrive():
        print("metadrive not installed, skipping")


def _maybe_resolve_env_class_spec(env_name: str) -> Optional[str]:
    if env_name in _METADRIVE_SHORTCUTS:
        return _METADRIVE_SHORTCUTS[env_name]
    if env_name.startswith("metadrive/"):
        class_name = env_name.split("/", 1)[1]
        return f"metadrive.envs:{class_name}"
    if ":" in env_name:
        return env_name
    if env_name.startswith("metadrive.") and env_name.count(".") >= 2:
        return env_name
    return None


def _normalize_env_kwargs_for_backend(env_name: str, env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(env_kwargs)
    is_metadrive_target = env_name.startswith("metadrive") or env_name.startswith("MetaDrive")
    if is_metadrive_target and isinstance(normalized.get("traffic_mode"), str):
        # MetaDrive traffic mode constants are lowercase strings.
        normalized["traffic_mode"] = normalized["traffic_mode"].lower()
    return normalized


def _resolve_class_from_spec(class_spec: str) -> Type[Any]:
    if ":" in class_spec:
        module_name, class_name = class_spec.split(":", 1)
    else:
        module_name, class_name = class_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    env_class = getattr(module, class_name)
    if class_name.startswith("MultiAgent"):
        raise ValueError(
            f"{class_name} is a multi-agent environment. "
            "Current DIME code expects a single-agent environment."
        )
    return env_class


def _make_env_from_class_spec(class_spec: str, env_kwargs: Dict[str, Any]):
    env_class = _resolve_class_from_spec(class_spec)
    # MetaDrive env classes typically take one config dict argument.
    try:
        return env_class(config=env_kwargs)
    except TypeError:
        try:
            return env_class(env_kwargs)
        except TypeError:
            return env_class(**env_kwargs)


def create_env(env_name: str, env_kwargs: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
    env_kwargs = _normalize_env_kwargs_for_backend(env_name, dict(env_kwargs or {}))
    try_import_env_registrations()

    class_spec = _maybe_resolve_env_class_spec(env_name)
    if class_spec is not None:
        return _make_env_from_class_spec(class_spec, env_kwargs)

    gym_make_kwargs = dict(env_kwargs)
    if render_mode is not None:
        gym_make_kwargs["render_mode"] = render_mode

    try:
        return gym.make(env_name, **gym_make_kwargs)
    except gym.error.NameNotFound as ex:
        if env_name.startswith("MetaDrive"):
            print("Gym ID not found. Fallback to metadrive/MetaDriveEnv.")
            return _make_env_from_class_spec("metadrive.envs:MetaDriveEnv", env_kwargs)
        raise ex
