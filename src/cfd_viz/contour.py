"""Contour plotting for structured CFD data."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public api: plot a filled contour of a scalar field on a structured grid
# --------------------------------------------------
def plot_contour(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    field_name: str = "field",
    levels: int = 50,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    filename: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a filled contour of a scalar field on a structured grid.

    Args:
        x: x-coordinates, shape (ni, nj) or (ni, nj, 1).
        y: y-coordinates, same shape as x.
        field: Scalar field data, same shape as x.
        field_name: Label for the colorbar.
        levels: Number of contour levels.
        figsize: Figure size (width, height) in inches.
        title: Plot title.  Defaults to the field name.
        filename: If given, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure.
    """
    # squeeze trailing singleton (nk=1) for 2-D datasets
    x = np.squeeze(x)
    y = np.squeeze(y)
    field = np.squeeze(field)

    ni, nj = x.shape

    # debug output for devs
    logger.debug("plot_contour: grid (%d x %d), field=%s, levels=%d", ni, nj, field_name, levels)

    # default title
    if title is None:
        title = field_name

    fig, ax = plt.subplots(figsize=figsize)

    # filled contour plot
    cf = ax.contourf(x, y, field, levels=levels, cmap="coolwarm")

    # colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(field_name)

    # set labels and formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

    fig.tight_layout()

    # save to file
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info("saved contour plot: %s", filename)

    # display
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
