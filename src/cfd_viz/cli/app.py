"""Typer application and command registration for cfd-viz."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import typer

from .callbacks import cli_callback
from .cmd_contour import cmd_contour
from .cmd_lst import lst_app
from .cmd_mesh import cmd_mesh

# --------------------------------------------------
# application and command groups
# --------------------------------------------------
app = typer.Typer(
    name="cfd-viz",
    help="Lightweight visualization CLI for structured CFD datasets.",
    no_args_is_help=True,
    add_completion=False,
)

# --------------------------------------------------
# register callbacks and commands
# --------------------------------------------------
app.callback()(cli_callback)
app.command(name="mesh")(cmd_mesh)
app.command(name="contour")(cmd_contour)
app.add_typer(lst_app, name="lst")


# --------------------------------------------------
# main entry point
# --------------------------------------------------
if __name__ == "__main__":
    app()
