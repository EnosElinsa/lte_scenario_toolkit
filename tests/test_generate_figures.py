from types import SimpleNamespace

import pandas as pd
import pytest

from lte_scenario_toolkit import generate_figures


def test_parser_requires_a_completed_run_and_exposes_no_csv_or_exact_output(capsys):
    with pytest.raises(SystemExit) as raised:
        generate_figures._parse_args(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--run-dir" in help_text
    assert "--csv" not in help_text
    assert "--config" not in help_text
    assert "--output-dir" not in help_text
    assert "--rect-id" not in help_text
    assert "legacy" not in help_text.lower()


def test_main_renders_one_unique_figure_run(monkeypatch, tmp_path, capsys):
    source_dir = tmp_path / "runs" / "city" / "profile" / "run"
    source_dir.mkdir(parents=True)
    source = SimpleNamespace(
        path=source_dir,
        scenario_id="city",
        profile_id="profile",
        run_id="a" * 32,
        rectangle={"center_x": 10.0, "center_y": 20.0, "pt_count": 1},
        frame=pd.DataFrame({"elevation": [12.5]}),
    )
    published = tmp_path / "published"
    captured = {}

    monkeypatch.setattr(
        generate_figures.FigureService,
        "load_source",
        staticmethod(lambda path: source),
    )

    def render(source_value, spec, service, formats, **kwargs):
        captured.update(
            source=source_value,
            output_root=service.output_root,
            formats=formats,
            parent_run_id=kwargs["parent_run_id"],
        )
        return published

    monkeypatch.setattr(
        generate_figures.FigureService,
        "render",
        staticmethod(render),
    )

    code = generate_figures.main(
        [
            "--run-dir",
            str(source_dir),
            "--output-root",
            str(tmp_path / "figure-runs"),
            "--format",
            "png",
        ]
    )

    assert code == 0
    assert captured["source"] is source
    assert captured["output_root"] == (tmp_path / "figure-runs").resolve()
    assert captured["formats"] == ("png",)
    assert captured["parent_run_id"] == "a" * 32
    assert f"Figure run: {published}" in capsys.readouterr().out


def test_main_maps_invalid_run_to_exit_code_two(monkeypatch, tmp_path, capsys):
    def fail(_path):
        raise ValueError("Figure source must be a completed run directory or run.json")

    monkeypatch.setattr(
        generate_figures.FigureService,
        "load_source",
        staticmethod(fail),
    )

    code = generate_figures.main(["--run-dir", str(tmp_path / "scenario.csv")])

    assert code == 2
    assert "completed run" in capsys.readouterr().err


def test_required_columns_remain_the_figure_service_contract():
    assert {"rect_id", "pt_count", "X", "Y"} <= generate_figures.REQUIRED_COLUMNS
