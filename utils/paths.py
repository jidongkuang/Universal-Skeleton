from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_like: str | os.PathLike | None, base: Path | None = None) -> Path | None:
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    anchor = base or PROJECT_ROOT
    return anchor / path


def env_path(env_name: str, default: str | os.PathLike) -> Path:
    value = os.environ.get(env_name)
    if value:
        return resolve_path(value)
    return resolve_path(default)


def env_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    return default if value is None else int(value)
