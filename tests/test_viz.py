"""Tests for cfd-viz plotting functions."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from cfd_viz.cli import app
from cfd_viz.contour import plot_contour
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

    # write Tecplot POINT format (i-fastest, then j, then k)
    out = tmp_path / "lst_table.dat"
    with open(out, "w", encoding="utf-8") as fobj:
        fobj.write('TITLE = "synthetic lst"\n')
        fobj.write('VARIABLES = "s", "freq.", "beta", "-im(alpha)"\n')
        fobj.write(f"ZONE I={ni}, J={nj}, K={nk}, F=POINT\n")
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    fobj.write(
                        f"{s[i, j, k]:.9e} {freq[i, j, k]:.9e} "
                        f"{beta[i, j, k]:.9e} {im_alpha[i, j, k]:.9e}\n"
                    )

    return out


class TestLSTCLI:
    """Tests for cfd-viz lst contour command."""

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

        def _fake_show() -> None:
            called["n"] += 1

        monkeypatch.setattr("cfd_viz.lst.plt.show", _fake_show)

        result = runner.invoke(
            app,
            [
                "lst",
                "contours",
                str(tecplot_lst_dat),
                "--single-k",
                "--k-index",
                "1",
                "--show",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert called["n"] == 1
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
                "--show",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "warning: --show ignored with --all-k" in result.output
        assert (out_dir / "alpi_kc_0000.png").exists()
