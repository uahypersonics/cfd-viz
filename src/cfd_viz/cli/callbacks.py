"""Shared callbacks for the cfd-viz command-line interface."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from importlib.metadata import version
from typing import Annotated

import typer


# --------------------------------------------------
# version callback
# --------------------------------------------------
def version_callback(value: bool) -> None:
    """Print the installed cfd-viz version and exit."""

    if value:
        typer.echo(f"cfd-viz {version('cfd-viz')}")
        raise typer.Exit()


# --------------------------------------------------
# verbose callback
# --------------------------------------------------
def verbose_callback(value: bool) -> None:
    """Configure console logging; --verbose adds debug output from this tool.

    Other packages (cfd-io, matplotlib, h5py, ...) stay at the default level,
    so --verbose shows this tool's debug messages without theirs.
    """

    # default console logging for every logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # debug output from this tool only; the packages it calls keep their levels
    if value:
        logging.getLogger("cfd_viz").setLevel(logging.DEBUG)


# --------------------------------------------------
# root application callback
# --------------------------------------------------
def cli_callback(
    version_requested: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose (debug) output.",
            callback=verbose_callback,
        ),
    ] = False,
) -> None:
    """cfd-viz: lightweight CFD visualization."""

    del version_requested, verbose
