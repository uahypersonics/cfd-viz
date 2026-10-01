"""Tests for cfd-viz plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from typer.testing import CliRunner

from cfd_viz.cli import app
from cfd_viz.contour import plot_contour
from cfd_viz.lst import (
    LSTFieldConfig,
    LSTPlotConfig,
    render_configured_lst_collection,
    render_standard_lst_contours,
)
from cfd_viz.mesh import plot_mesh

# --------------------------------------------------
# test data
# --------------------------------------------------
NI, NJ = 11, 9

# simple rectangular grid
X, Y = np.meshgrid(
    np.linspace(0.0, 1.0, NI),
    np.linspace(0.0, 0.5, NJ),
    indexing="ij",
)

runner = CliRunner()


# --------------------------------------------------
# mesh tests
# --------------------------------------------------
class TestPlotMesh:
    """Tests for the mesh plotting function."""

    def test_returns_figure(self):
        fig = plot_mesh(X, Y, show=False)
        assert fig is not None

    def test_skip(self):
        """Skip parameter should not raise."""
        fig = plot_mesh(X, Y, skip=3, show=False)
        assert fig is not None

    def test_3d_squeeze(self):
        """Handles (ni, nj, 1) arrays by squeezing."""
        x3 = X[:, :, np.newaxis]
        y3 = Y[:, :, np.newaxis]
        fig = plot_mesh(x3, y3, show=False)
        assert fig is not None

    def test_save(self, tmp_path):
        """Save to file."""
        out = tmp_path / "mesh.png"
        plot_mesh(X, Y, filename=str(out), show=False)
        assert out.exists()
        assert out.stat().st_size > 0


# --------------------------------------------------
# contour tests
# --------------------------------------------------
class TestPlotContour:
    """Tests for the contour plotting function."""

    def test_returns_figure(self):
        field = np.sin(np.pi * X) * np.cos(np.pi * Y)
        fig = plot_contour(X, Y, field, show=False)
        assert fig is not None

    def test_3d_squeeze(self):
        """Handles (ni, nj, 1) arrays by squeezing."""
        x3 = X[:, :, np.newaxis]
        y3 = Y[:, :, np.newaxis]
        f3 = np.ones_like(x3)
        fig = plot_contour(x3, y3, f3, show=False)
        assert fig is not None

    def test_save(self, tmp_path):
        """Save to file."""
        out = tmp_path / "contour.png"
        field = X + Y
        plot_contour(X, Y, field, filename=str(out), show=False)
        assert out.exists()
        assert out.stat().st_size > 0


# --------------------------------------------------
# CLI tests
# --------------------------------------------------

# small synthetic grid for CLI integration tests
NI_CLI, NJ_CLI, NK_CLI = 20, 15, 1


@pytest.fixture()
def h5_grid(tmp_path):
    """Write a small grid-only HDF5 file and return its path."""
    from cfd_io import write_file
    from cfd_io.dataset import Dataset, StructuredGrid

    # build a simple rectangular grid
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, NI_CLI),
        np.linspace(0.0, 0.5, NJ_CLI),
        indexing="ij",
    )
    z = np.zeros_like(x)

    # add trailing nk=1 dimension
    grid = StructuredGrid(
        x=x[:, :, np.newaxis],
        y=y[:, :, np.newaxis],
        z=z[:, :, np.newaxis],
    )
    ds = Dataset(grid=grid)

    out = tmp_path / "grid.h5"
    write_file(str(out), ds)
    return out


@pytest.fixture()
def h5_flow(tmp_path):
    """Write a small grid+flow HDF5 file and return its path."""
    from cfd_io import write_file
    from cfd_io.dataset import Dataset, Field, StructuredGrid

    # build a simple rectangular grid
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, NI_CLI),
        np.linspace(0.0, 0.5, NJ_CLI),
        indexing="ij",
    )
    z = np.zeros_like(x)

    # add trailing nk=1 dimension
    shape = (NI_CLI, NJ_CLI, NK_CLI)
    grid = StructuredGrid(
        x=x[:, :, np.newaxis],
        y=y[:, :, np.newaxis],
        z=z[:, :, np.newaxis],
    )

    # synthetic flow field
    flow = {"pres": Field(data=np.random.rand(*shape))}

    ds = Dataset(grid=grid, flow=flow)

    out = tmp_path / "flow.h5"
    write_file(str(out), ds)
    return out


class TestCLI:
    """Tests for the CLI entry points."""

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "cfd-viz" in result.output

    def test_no_args(self):
        result = runner.invoke(app, [])
        # typer returns exit code 0 (help displayed)
        assert result.exit_code in (0, 2)

    def test_mesh_missing_file(self):
        result = runner.invoke(app, ["mesh", "nonexistent.su2"])
        assert result.exit_code != 0

    def test_contour_missing_file(self):
        result = runner.invoke(app, ["contour", "nonexistent.h5", "--field", "pres"])
        assert result.exit_code != 0

    def test_mesh_from_h5(self, h5_grid, tmp_path):
        """CLI mesh reads an HDF5 grid and saves a figure."""
        out = tmp_path / "mesh.png"
        result = runner.invoke(app, ["mesh", str(h5_grid), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_mesh_with_skip(self, h5_grid, tmp_path):
        """CLI mesh with --skip flag."""
        out = tmp_path / "mesh_skip.png"
        result = runner.invoke(app, ["mesh", str(h5_grid), "--skip", "3", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_contour_from_h5(self, h5_flow, tmp_path):
        """CLI contour reads an HDF5 file and plots a field."""
        out = tmp_path / "pres.png"
        result = runner.invoke(app, ["contour", str(h5_flow), "--field", "pres", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_contour_bad_field(self, h5_flow):
        """CLI contour with nonexistent field name."""
        result = runner.invoke(app, ["contour", str(h5_flow), "--field", "bogus"])
        assert result.exit_code != 0
        assert "bogus" in result.output


@pytest.fixture()
def tecplot_lst_dat(tmp_path):
    """Write a synthetic LST-style Tecplot ASCII table file."""
    ni, nj, nk = 6, 5, 3

    # build simple rectilinear LST axes
    s_vec = np.linspace(0.1, 0.6, ni)
    freq_vec = np.linspace(100000.0, 300000.0, nj)
    beta_vec = np.array([0.0, 5.0, 10.0])

    # assemble arrays with shape (ni, nj, nk)
    s = np.repeat(s_vec[:, np.newaxis, np.newaxis], nj, axis=1)
    s = np.repeat(s, nk, axis=2)

    freq = np.repeat(freq_vec[np.newaxis, :, np.newaxis], ni, axis=0)
    freq = np.repeat(freq, nk, axis=2)

    beta = np.repeat(beta_vec[np.newaxis, np.newaxis, :], ni, axis=0)
    beta = np.repeat(beta, nj, axis=1)

    # synthetic positive field so positive-rounded policy is stable
    im_alpha = 10.0 + 20.0 * np.exp(-3.0 * s) + 0.00001 * freq + 0.1 * beta
    nfac3 = 0.5 * im_alpha

    # write Tecplot POINT format (i-fastest, then j, then k)
    out = tmp_path / "growth_rate_with_nfact_amps.dat"
    with open(out, "w", encoding="utf-8") as fobj:
        fobj.write('TITLE = "synthetic lst"\n')
        fobj.write('VARIABLES = "s", "freq.", "beta", "-im(alpha)", "Nfac3"\n')
        fobj.write(f"ZONE I={ni}, J={nj}, K={nk}, F=POINT\n")
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    fobj.write(
                        f"{s[i, j, k]:.9e} {freq[i, j, k]:.9e} "
                        f"{beta[i, j, k]:.9e} {im_alpha[i, j, k]:.9e} "
                        f"{nfac3[i, j, k]:.9e}\n"
                    )

    return out


class TestLSTCLI:
    """Tests for cfd-viz lst contour command."""

    def test_lst_init_writes_default_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["lst", "init"])

        config_path = tmp_path / "cfd-viz-lst.toml"
        assert result.exit_code == 0
        assert config_path.exists()
        config_text = config_path.read_text(encoding="utf-8")
        assert "[contour.alpi]" in config_text
        assert "[contour.nfac]" in config_text
        assert "contour_var" not in config_text
        assert "#--------------------------------------------------\n# input" in config_text
        assert "[[fields]]" not in config_text

    def test_lst_init_requires_force_to_replace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "cfd-viz-lst.toml"
        config_path.write_text("custom", encoding="utf-8")

        refused = runner.invoke(app, ["lst", "init"])
        replaced = runner.invoke(app, ["lst", "init", "--force"])

        assert refused.exit_code == 1
        assert "use --force to replace it" in refused.output
        assert replaced.exit_code == 0
        assert "[contour]" in config_path.read_text(encoding="utf-8")

    def test_lst_contours_discovers_local_config(self, tecplot_lst_dat, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "cfd-viz-lst.toml"
        config_path.write_text(
            """
[input]
path = "growth_rate_with_nfact_amps.dat"
dimensions = 3
x_var = "s"
y_var = "freq,freq."
z_var = "beta"

[contour.nfac]
variable = "Nfac3"

[output]
directory = "configured"
dpi = 120

[z_axis]
all_planes = false
index = 2
show_label = true

[style]
use_tex = false

""".strip(),
            encoding="utf-8",
        )

        result = runner.invoke(app, ["lst", "contours"])

        assert result.exit_code == 0, result.output
        assert "wrote 1 plot(s)" in result.output
        assert (tmp_path / "configured" / "nfac_kc_0005.png").exists()

    def test_lst_contours_cli_values_override_explicit_config(self, tecplot_lst_dat, tmp_path):
        config_dir = tmp_path / "settings"
        config_dir.mkdir()
        config_path = config_dir / "custom.toml"
        config_path.write_text(
            """
[input]
path = "../growth_rate_with_nfact_amps.dat"
dimensions = 3
x_var = "s"
y_var = "freq,freq."
z_var = "beta"

[contour.nfac]
variable = "Nfac3"

[output]
directory = "configured"

[z_axis]
all_planes = false
index = 2

[style]
use_tex = false

""".strip(),
            encoding="utf-8",
        )
        override_dir = tmp_path / "override"

        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                "--config",
                str(config_path),
                "--k-index",
                "1",
                "--out-dir",
                str(override_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (override_dir / "nfac_kc_0000.png").exists()

    def test_lst_contours_renders_two_dimensional_input(self, tmp_path, monkeypatch):
        data_path = tmp_path / "lst_2d.dat"
        data_path.write_text(
            """
TITLE = "synthetic 2d lst"
VARIABLES = "s", "freq.", "Nfac3"
ZONE I=3, J=3, F=POINT
0.0 1000.0 0.0
0.5 1000.0 1.0
1.0 1000.0 2.0
0.0 2000.0 1.0
0.5 2000.0 2.0
1.0 2000.0 3.0
0.0 3000.0 2.0
0.5 3000.0 3.0
1.0 3000.0 4.0
""".strip(),
            encoding="utf-8",
        )
        config_path = tmp_path / "cfd-viz-lst.toml"
        config_path.write_text(
            """
[input]
path = "lst_2d.dat"
dimensions = 2
x_var = "s"
y_var = "freq,freq."

[contour.nfac]
variable = "Nfac3"

[z_axis]
show_label = false

[style]
use_tex = false
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["lst", "contours"])

        assert result.exit_code == 0, result.output
        assert "wrote 1 plot(s)" in result.output
        assert (tmp_path / "nfac_kc.png").exists()

    def test_standard_lst_api_uses_default_file(self, tecplot_lst_dat, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        files = render_standard_lst_contours(out_dir="api_plots", all_k=False, k_index=1)

        assert [path.name for path in files] == [
            "alpi_kc_0000.png",
            "nfac_kc_0000.png",
        ]

    def test_lst_collection_renders_files_with_shared_field_bounds(
        self, tecplot_lst_dat, tmp_path
    ):
        config = LSTPlotConfig(
            fields=(
                LSTFieldConfig(
                    name="-im(alpha)",
                    prefix="alpi_kc",
                    label=r"$-\alpha_i$ [1/m]",
                ),
            ),
            output_dir=tmp_path / "collection",
            all_k=False,
            use_tex=False,
        )

        files = render_configured_lst_collection(
            config,
            [tecplot_lst_dat, tecplot_lst_dat],
            prefix_suffixes=["first", "second"],
            single_plane=True,
        )

        assert [path.name for path in files] == [
            "alpi_kc_first.png",
            "alpi_kc_second.png",
        ]

    def test_lst_contours_defaults_to_standard_file_and_fields(
        self, tecplot_lst_dat, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["lst", "contours"])

        assert result.exit_code == 0
        assert "wrote 6 plot(s)" in result.output
        assert (tmp_path / "alpi_kc_0000.png").exists()
        assert (tmp_path / "nfac_kc_0000.png").exists()

    def test_lst_contours_all_k(self, tecplot_lst_dat, tmp_path):
        out_dir = tmp_path / "plots"
        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                str(tecplot_lst_dat),
                "--all-k",
                "--out-dir",
                str(out_dir),
                "--prefix",
                "alpi_kc",
                "--field",
                "-im(alpha)",
            ],
        )

        assert result.exit_code == 0
        assert "wrote 3 plot(s)" in result.output
        assert (out_dir / "alpi_kc_0000.png").exists()
        assert (out_dir / "alpi_kc_0005.png").exists()
        assert (out_dir / "alpi_kc_0010.png").exists()

    def test_lst_contours_single_k(self, tecplot_lst_dat, tmp_path):
        out_dir = tmp_path / "single"
        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                str(tecplot_lst_dat),
                "--single-k",
                "--k-index",
                "2",
                "--out-dir",
                str(out_dir),
                "--prefix",
                "alpi_kc",
                "--field",
                "-im(alpha)",
            ],
        )

        assert result.exit_code == 0
        assert "wrote 1 plot(s)" in result.output
        assert (out_dir / "alpi_kc_0005.png").exists()

    def test_lst_contours_single_k_show_calls_matplotlib_show(
        self,
        tecplot_lst_dat,
        tmp_path,
        monkeypatch,
    ):
        out_dir = tmp_path / "show_single"
        called = {"n": 0}
        rendered = {}

        def _fake_show() -> None:
            called["n"] += 1
            figure = plt.gcf()
            plot_axis = figure.axes[0]
            colorbar_axis = figure.axes[-1]
            rendered["colorbar_ticks"] = colorbar_axis.get_yticks()
            rendered["outline_visible"] = all(
                spine.get_visible() for spine in plot_axis.spines.values()
            )
            rendered["title"] = plot_axis.get_title()

        monkeypatch.setattr("cfd_viz.lst.plt.show", _fake_show)

        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                str(tecplot_lst_dat),
                "--single-k",
                "--field",
                "-im(alpha)",
                "--k-index",
                "1",
                "--show",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert called["n"] == 1
        assert len(rendered["colorbar_ticks"]) == 2
        assert rendered["outline_visible"] is True
        assert rendered["title"] == r"$k_c$ = 0"
        assert (out_dir / "alpi_kc_0000.png").exists()

    def test_lst_contours_all_k_show_emits_warning(self, tecplot_lst_dat, tmp_path):
        out_dir = tmp_path / "show_all"
        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                str(tecplot_lst_dat),
                "--all-k",
                "--field",
                "-im(alpha)",
                "--show",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "warning: --show ignored with --all-k" in result.output
        assert (out_dir / "alpi_kc_0000.png").exists()
