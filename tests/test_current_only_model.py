import json
from pathlib import Path

import pytest
import yaml

import lte_scenario_toolkit.candidate_cache as candidate_cache
import lte_scenario_toolkit.profiles as profiles
from lte_scenario_toolkit.figure_service import FigureService
from lte_scenario_toolkit.gui.settings import GuiSettingsStore
from lte_scenario_toolkit.run_service import RunService
from lte_scenario_toolkit.select_sites import _parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_profile_model_exposes_only_the_current_unversioned_shape():
    assert "schema_version" not in profiles.ExperimentProfile.__dataclass_fields__
    assert not hasattr(profiles, "LegacyProfileValues")
    assert not hasattr(profiles, "load_legacy_profile")
    assert not hasattr(profiles, "convert_legacy_profile")


def test_candidate_cache_exposes_only_the_current_unversioned_shape():
    assert not hasattr(candidate_cache, "CACHE_SCHEMA_VERSION")
    assert not hasattr(candidate_cache, "legacy_cache_filename")
    assert not hasattr(candidate_cache.CandidateCache, "import_legacy")


def test_gui_settings_are_written_without_a_schema_discriminator(tmp_path):
    store = GuiSettingsStore(tmp_path)
    store.path.parent.mkdir(parents=True)
    store.path.write_text(
        json.dumps(
            {"schema_version": 1, "language": "zh-CN", "output_roots": []}
        ),
        encoding="utf-8",
    )

    assert store.load().language == "en"
    store.save(language="en", output_roots=[])

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload == {"language": "en", "output_roots": []}


def test_figures_reject_a_bare_csv_source_and_have_no_dem_attachment(tmp_path):
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text("rect_id,pt_count\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completed run"):
        FigureService.load_source(csv_path)
    assert not hasattr(FigureService, "attach_dem")
    assert not hasattr(FigureService, "inspect_source")


def test_selection_cli_has_one_web_selector_surface(capsys):
    with pytest.raises(SystemExit) as raised:
        _parse_args(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--select-index" in help_text
    assert "--selector" not in help_text
    assert "legacy" not in help_text.lower()


def test_run_service_has_no_exact_compatibility_relocation():
    assert not hasattr(RunService, "relocate_to_exact_directory")


def test_repository_catalog_manifest_and_profiles_are_unversioned():
    catalog = yaml.safe_load((ROOT / "data" / "datasets.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert "schema_version" not in catalog
    assert "schema_version" not in manifest

    for path in (ROOT / "configs" / "example.yaml", ROOT / "configs" / "newyork.yaml"):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "schema_version" not in profile
        assert {"profile", "inputs", "experiment", "spatial", "scan", "outputs", "figures"} <= set(profile)
