"""Render filled contours from CFD datasets."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer


# --------------------------------------------------
# contour command
# --------------------------------------------------
def cmd_contour(
    path: Annotated[
        Path,
        typer.Argument(help="Path to a CFD data file supported by cfd-io."),
    ],
    field: Annotated[
        str,
        typer.Option("--field", "-f", help="Flow variable to plot."),
    ],
    grid_in: Annotated[
        str | None,
        typer.Option("--grid-in", "-g", help="Separate grid file for split formats."),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Save the figure to a file."),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option("--title", "-t", help="Plot title."),
    ] = None,
    levels: Annotated[
        int,
        typer.Option("--levels", "-l", help="Number of contour levels."),
    ] = 50,
) -> None:
    """Display a filled contour of a scalar field."""

    from cfd_io import read_file

    from cfd_viz.contour import plot_contour

    # read the dataset and validate the selected field
    dataset = read_file(str(path), grid_file=grid_in)
    available_fields = list(dataset.flow.keys())
    if field not in dataset.flow:
        typer.echo(
            f"Field '{field}' not found. Available fields: {available_fields}",
            err=True,
        )
        raise typer.Exit(code=1)

    # plot the field and display only when no output path was requested
    plot_contour(
        dataset.grid.x,
        dataset.grid.y,
        dataset.flow[field].data,
        field_name=field,
        levels=levels,
        title=title,
        filename=output,
        show=output is None,
    )
