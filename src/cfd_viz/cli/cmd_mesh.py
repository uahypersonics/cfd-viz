"""Render structured CFD meshes."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


# --------------------------------------------------
# mesh command
# --------------------------------------------------
def cmd_mesh(
    path: Annotated[
        Path,
        typer.Argument(help="Path to a CFD grid file supported by cfd-io."),
    ],
    grid_in: Annotated[
        str | None,
        typer.Option("--grid-in", "-g", help="Separate grid file for split formats."),
    ] = None,
    skip: Annotated[
        int,
        typer.Option("--skip", "-s", help="Plot every N-th grid line."),
    ] = 1,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Save the figure to a file."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Plot title."),
    ] = "Mesh",
) -> None:
    """Display a structured mesh."""

    from cfd_io import read_file

    from cfd_viz.mesh import plot_mesh

    # default grid file to the input path for grid-only split files
    grid_path = str(path) if grid_in is None else grid_in

    # read the dataset and validate its grid
    dataset = read_file(str(path), grid_file=grid_path)
    if dataset.grid is None:
        typer.echo("No grid data found in the file.", err=True)
        raise typer.Exit(code=1)

    # plot the grid and display only when no output path was requested
    plot_mesh(
        dataset.grid.x,
        dataset.grid.y,
        skip=skip,
        title=title,
        filename=output,
        show=output is None,
    )
