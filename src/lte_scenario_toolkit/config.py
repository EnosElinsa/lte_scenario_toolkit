"""Load reproducible experiment configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .profiles import ExperimentProfile, _profile_repository, load_profile


class ExperimentConfig(dict[str, Any]):
    """Runtime mapping that retains one parsed profile snapshot."""

    def __init__(
        self,
        *args: Any,
        profile_snapshot: ExperimentProfile | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.profile_snapshot = profile_snapshot


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _infer_project_root(config_path: Path) -> Path:
    return _profile_repository(config_path)


def load_experiment_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    city: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return one validated profile in the runtime service mapping.

    Relative paths resolve against the repository root so the same profile
    behaves consistently in CI and when invoked from another directory.
    """

    path = Path(config_path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _infer_project_root(path)
    )
    profile = load_profile(path, repo_root=root)
    if city is not None:
        resolved_city = city
        catalog_path = root / "data" / "datasets.yaml"
        if catalog_path.is_file():
            from .data_catalog import load_data_catalog

            catalog = load_data_catalog(catalog_path, repo_root=root)
            if city not in catalog.scenarios_by_id:
                matches = [
                    scenario_id
                    for scenario_id, scenario in catalog.scenarios_by_id.items()
                    if str(scenario["display_name"]).casefold() == city.casefold()
                ]
                if len(matches) == 1:
                    resolved_city = matches[0]
        if resolved_city != profile.scenario_id:
            raise ValueError(
                f"--city {city!r} does not match profile scenario_id "
                f"{profile.scenario_id!r}"
            )

    config = ExperimentConfig(profile.runtime_values(), profile_snapshot=profile)
    config["repo_root"] = root
    if output_dir is not None:
        config["output_root"] = _resolve_path(output_dir, root)
    return config
