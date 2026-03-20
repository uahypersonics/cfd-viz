"""Structured mesh plotting."""

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
# public api: plot a structured mesh as grid lines
# --------------------------------------------------
def plot_mesh(
    x: np.ndarray,
    y: np.ndarray,
    *,
    skip: int = 1,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Mesh",
    filename: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a 2-D structured mesh as grid lines.

    Args:
        x: x-coordinates, shape (ni, nj) or (ni, nj, 1).
        y: y-coordinates, same shape as x.
        skip: Plot every *skip*-th grid line in each direction.
            Useful for dense meshes where plotting every line is slow.
        figsize: Figure size (width, height) in inches.
        title: Plot title.
        filename: If given, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure.
    """
    # squeeze trailing singleton (nk=1) for 2-D datasets
    x = np.squeeze(x)
    y = np.squeeze(y)

    ni, nj = x.shape

    # debug output for devs
    logger.debug("plot_mesh: grid (%d x %d), skip=%d", ni, nj, skip)

    fig, ax = plt.subplots(figsize=figsize)

    # draw grid lines in the i-direction
    for i in range(0, ni, skip):
        ax.plot(x[i, :], y[i, :], color="steelblue", linewidth=0.4, alpha=0.7)

    # draw grid lines in the j-direction
    for j in range(0, nj, skip):
        ax.plot(x[:, j], y[:, j], color="steelblue", linewidth=0.4, alpha=0.7)

    # set labels and formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

    fig.tight_layout()

    # save to file
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info("saved mesh plot: %s", filename)

    # display
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
