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
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging."),
    ] = False,
) -> None:
    """cfd-viz: lightweight CFD visualization."""

    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(levelname)-8s %(message)s",
    )
