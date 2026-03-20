"""cfd-viz command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

# --------------------------------------------------
# app
# --------------------------------------------------
app = typer.Typer(
    name="cfd-viz",
    help="Lightweight visualization CLI for structured CFD datasets.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _configure_logging(debug: bool) -> None:
    """Set up console logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(message)s",
    )


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from importlib.metadata import version

        typer.echo(f"cfd-viz {version('cfd-viz')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """cfd-viz: lightweight CFD visualization."""
    # set up logging
    _configure_logging(debug)


# --------------------------------------------------
# mesh subcommand
# --------------------------------------------------
@app.command()
def mesh(
    path: Path = typer.Argument(..., help="Path to a CFD grid file (any format cfd-io supports)."),
    grid_in: str | None = typer.Option(None, "--grid-in", "-g", help="Separate grid file (for split formats)."),
    skip: int = typer.Option(1, "--skip", "-s", help="Plot every N-th grid line."),
    output: str | None = typer.Option(None, "--output", "-o", help="Save figure to file."),
    title: str = typer.Option("Mesh", "--title", "-t", help="Plot title."),
) -> None:
    """Display a structured mesh."""
    from cfd_io import read_file

    from cfd_viz.mesh import plot_mesh

    # default grid file to the input path (handles grid-only split files)
    if grid_in is None:
        grid_in = str(path)

    # read the dataset
    ds = read_file(str(path), grid_file=grid_in)

    # validate grid data exists
    if ds.grid is None:
        typer.echo("No grid data found in the file.", err=True)
        raise typer.Exit(code=1)

    # extract 2-D grid arrays
    x = ds.grid.x
    y = ds.grid.y

    # plot (show interactively only when not saving to file)
    plot_mesh(x, y, skip=skip, title=title, filename=output, show=output is None)


# --------------------------------------------------
# contour subcommand
# --------------------------------------------------
@app.command()
def contour(
    path: Path = typer.Argument(..., help="Path to a CFD data file (any format cfd-io supports)."),
    field: str = typer.Option(..., "--field", "-f", help="Name of the flow variable to plot."),
    grid_in: str | None = typer.Option(None, "--grid-in", "-g", help="Separate grid file (for split formats)."),
    output: str | None = typer.Option(None, "--output", "-o", help="Save figure to file."),
    title: str | None = typer.Option(None, "--title", "-t", help="Plot title."),
    levels: int = typer.Option(50, "--levels", "-l", help="Number of contour levels."),
) -> None:
    """Display a filled contour of a scalar field."""
    from cfd_io import read_file

    from cfd_viz.contour import plot_contour

    # read the dataset
    ds = read_file(str(path), grid_file=grid_in)

    # validate that the requested field exists
    available = list(ds.flow.keys())
    if field not in ds.flow:
        typer.echo(f"Field '{field}' not found. Available fields: {available}", err=True)
        raise typer.Exit(code=1)

    # extract arrays
    x = ds.grid.x
    y = ds.grid.y
    data = ds.flow[field].data

    # plot (show interactively only when not saving to file)
    plot_contour(x, y, data, field_name=field, levels=levels, title=title, filename=output, show=output is None)
