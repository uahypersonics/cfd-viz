"""Tests for LST plotting configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfd_viz.lst import load_lst_config, write_default_lst_config


def test_default_lst_config_round_trip(tmp_path: Path) -> None:
    """Generated defaults must load with paths relative to the config file."""

    config_path = tmp_path / "cfd-viz-lst.toml"

    written = write_default_lst_config(config_path)
    config = load_lst_config(config_path)

    assert written == config_path
    assert config.input_path == tmp_path / "growth_rate_with_nfact_amps.dat"
    assert config.output_dir == tmp_path
    assert config.figure_width == 3.25
    assert config.figure_height == 2.5
    assert config.colormap == "jet"
    assert config.font_size == 11.0
    assert config.dimensions == 3
    assert config.xvar == "s"
    assert config.yvar == "freq,freq."
    assert config.kvar == "beta"
    assert [field.name for field in config.fields] == ["-im(alpha)", "Nfac3"]
    assert [field.label for field in config.fields] == [r"$-\alpha_i$ [1/m]", r"$N$"]
    assert [field.levels_count for field in config.fields] == [60, 60]

    config_text = config_path.read_text(encoding="utf-8")
    assert "#--------------------------------------------------\n# input" in config_text
    assert "contour_var" not in config_text
    assert "[x_axis]" in config_text
    assert "[y_axis]" in config_text
    assert "[z_axis]" in config_text
    assert "[contour.alpi]" in config_text
    assert "[contour.nfac]" in config_text
    assert "[[fields]]" not in config_text


def test_write_default_lst_config_requires_force(tmp_path: Path) -> None:
    """Existing configuration files must not be replaced accidentally."""

    config_path = tmp_path / "cfd-viz-lst.toml"
    config_path.write_text("custom", encoding="utf-8")

    with pytest.raises(FileExistsError, match="use --force to replace it"):
        write_default_lst_config(config_path)

    write_default_lst_config(config_path, force=True)

    assert "[style]" in config_path.read_text(encoding="utf-8")


def test_load_lst_config_reads_custom_plot_settings(tmp_path: Path) -> None:
    """Configured fields and presentation settings must override defaults."""

    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[input]
path = "results.dat"
dimensions = 2
x_var = "station"
y_var = "frequency"

[output]
directory = "figures"
dpi = 600

[contour]

[contour.nfac]
variable = "Nfac2"
levels_policy = "global-auto"
levels_count = 25
level_min = -2.0
level_max = 8.0

[x_axis]
label = "Station"
scale = 2.0
ticks = 4

[y_axis]
label = "Frequency"
scale = 0.5
ticks = 5

[z_axis]
show_label = false

[style]
figure_width = 6.5
figure_height = 4.0
colormap = "plasma"
font_size = 9.0
use_tex = false
""".strip(),
        encoding="utf-8",
    )

    config = load_lst_config(config_path)

    assert config.input_path == tmp_path / "results.dat"
    assert config.output_dir == tmp_path / "figures"
    assert config.dpi == 600
    assert config.fields[0].level_min == -2.0
    assert config.fields[0].level_max == 8.0
    assert config.figure_width == 6.5
    assert config.colormap == "plasma"
    assert config.dimensions == 2
    assert config.xvar == "station"
    assert config.yvar == "frequency"
    assert config.x_ticks == 4
    assert config.y_ticks == 5
    assert config.fields[0].name == "Nfac2"
    assert config.fields[0].prefix == "nfac2_kc"


def test_load_lst_config_preserves_empty_input_for_workflow_discovery(
    tmp_path: Path,
) -> None:
    """An empty input path must remain unset for workflow-owned discovery."""

    config_path = tmp_path / "discover.toml"
    config_path.write_text('[input]\npath = ""\n', encoding="utf-8")

    config = load_lst_config(config_path)

    assert config.input_path is None


def test_load_lst_config_rejects_one_sided_contour_bounds(tmp_path: Path) -> None:
    """A partial explicit contour range must produce a configuration error."""

    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[contour]
level_min = 0.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="level_min and level_max must be set together"):
        load_lst_config(config_path)


def test_load_lst_config_reads_independent_contour_settings(tmp_path: Path) -> None:
    """Each selected contour variable must retain its own name and levels."""

    config_path = tmp_path / "contours.toml"
    config_path.write_text(
        r"""
[input]

[contour.alpi]
variable = "-im(alpha)"
name = "Growth rate"
output_prefix = "growth"
levels_policy = "global-auto"
levels_count = 31
level_min = 0.0
level_max = 7500.0

[contour.nfac]
variable = "Nfac3"
name = '$N$'
output_prefix = "n_factor"
levels_policy = "positive-rounded"
levels_count = 11
level_min = 0.0
level_max = 5.0
""".strip(),
        encoding="utf-8",
    )

    config = load_lst_config(config_path)

    growth_rate, n_factor = config.fields
    assert growth_rate.label == "Growth rate"
    assert growth_rate.prefix == "growth"
    assert growth_rate.levels_policy == "global-auto"
    assert growth_rate.levels_count == 31
    assert growth_rate.level_max == 7500.0
    assert n_factor.label == r"$N$"
    assert n_factor.prefix == "n_factor"
    assert n_factor.levels_count == 11
    assert n_factor.level_max == 5.0
