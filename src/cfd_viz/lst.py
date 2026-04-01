"""LST-specific contour plotting helpers.

This module handles flow-only LST Tecplot tables where grid coordinates are
stored as regular flow variables (for example s, freq., beta).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# parse comma-separated candidate names
# --------------------------------------------------
def _split_candidates(raw: str) -> list[str]:
    """Parse a comma-separated list into normalized candidate names."""
    return [name.strip() for name in raw.split(",") if name.strip()]


# --------------------------------------------------
# pick first matching flow variable
# --------------------------------------------------
def _pick_flow_var(flow: dict[str, object], raw_candidates: str) -> str:
    """Return first matching flow variable from a candidate list."""

    # parse candidates and check in order
    candidates = _split_candidates(raw_candidates)
    for name in candidates:
        if name in flow:
            return name

    # include available keys in error for user debugging
    available = list(flow.keys())
    raise KeyError(
        f"none of the requested variables were found: {candidates}; available={available}"
    )


# --------------------------------------------------
# compute shared contour levels
# --------------------------------------------------
def _build_levels(data: np.ndarray, policy: str, count: int) -> np.ndarray:
    """Build shared contour levels using a named policy."""

    if count < 2:
        raise ValueError("levels_count must be >= 2")

    # derive min/max bounds from selected policy
    if policy == "global-auto":
        level_min = float(np.min(data))
        level_max = float(np.max(data))
    elif policy == "positive-rounded":
        level_min = 0.0
        raw_max = float(np.max(data))
        level_max = float(math.ceil(raw_max / 10.0) * 10.0)
    else:
        raise ValueError(
            f"unknown levels policy '{policy}'. Use one of: global-auto, positive-rounded"
        )

    # avoid invalid or degenerate contour ranges
    if not np.isfinite(level_min) or not np.isfinite(level_max):
        raise ValueError("non-finite contour bounds detected")
    if level_max <= level_min:
        level_max = level_min + 1.0

    return np.linspace(level_min, level_max, count)


# --------------------------------------------------
# build contour levels from explicit bounds
# --------------------------------------------------
def _build_levels_from_bounds(
    *,
    level_min: float,
    level_max: float,
    count: int,
) -> np.ndarray:
    """Build contour levels from explicit min/max bounds."""

    if count < 2:
        raise ValueError("levels_count must be >= 2")
    if not np.isfinite(level_min) or not np.isfinite(level_max):
        raise ValueError("non-finite contour bounds detected")
    if level_max <= level_min:
        level_max = level_min + 1.0

    return np.linspace(level_min, level_max, count)


# --------------------------------------------------
# public API: render one or many LST contour plots
# --------------------------------------------------
def render_lst_contours(
    *,
    path: str | Path,
    field: str = "-im(alpha)",
    xvar: str = "s",
    yvar: str = "freq,freq.",
    kvar: str = "beta",
    all_k: bool = True,
    k_index: int = 1,
    out_dir: str | Path = ".",
    prefix: str = "alpi_kc",
    levels_policy: str = "positive-rounded",
    levels_count: int = 60,
    level_min_override: float | None = None,
    level_max_override: float | None = None,
    clip_below: bool = True,
    dpi: int = 300,
    show: bool = False,
) -> list[Path]:
    """Render LST contours from a flow-only table dataset.

    Args:
        path: Input file path readable by cfd-io.
        field: Flow variable name for contour values.
        xvar: Candidate names for x-axis variable (comma-separated).
        yvar: Candidate names for y-axis variable (comma-separated).
        kvar: Candidate names for k/beta variable (comma-separated).
        all_k: When True, render all k-planes.
        k_index: 1-based k index when all_k is False.
        out_dir: Directory to write PNG images.
        prefix: Output filename stem before beta token.
        levels_policy: Contour policy: global-auto or positive-rounded.
        levels_count: Number of contour levels.
        level_min_override: Optional explicit lower contour bound.
        level_max_override: Optional explicit upper contour bound.
        clip_below: Clip values below min(levels) to min(levels).
        dpi: PNG output resolution.
        show: Display figures interactively (recommended for single-k only).

    Returns:
        List of written PNG paths.
    """

    # read dataset from cfd-io
    from cfd_io import read_file

    ds = read_file(str(path))
    flow = ds.flow

    # resolve variable names from candidate aliases
    x_name = _pick_flow_var(flow, xvar)
    y_name = _pick_flow_var(flow, yvar)
    k_name = _pick_flow_var(flow, kvar)
    field_name = _pick_flow_var(flow, field)

    # extract mapped arrays
    x = flow[x_name].data
    y = flow[y_name].data
    z = flow[k_name].data
    data = flow[field_name].data

    # validate shape compatibility
    if x.shape != y.shape or x.shape != z.shape or x.shape != data.shape:
        raise ValueError(
            "mapped variables must have matching shapes: "
            f"x={x.shape}, y={y.shape}, z={z.shape}, field={data.shape}"
        )
    if data.ndim != 3:
        raise ValueError(f"expected 3-D data for k-slices, got shape={data.shape}")

    # build shared contour levels once for consistent comparisons
    if level_min_override is not None and level_max_override is not None:
        levels = _build_levels_from_bounds(
            level_min=float(level_min_override),
            level_max=float(level_max_override),
            count=levels_count,
        )
    else:
        levels = _build_levels(data, policy=levels_policy, count=levels_count)
    level_min = float(levels[0])
    level_max = float(levels[-1])

    # decide which k-planes to render
    nk = data.shape[2]
    if all_k:
        k_indices = list(range(nk))
    else:
        idx = k_index - 1
        if idx < 0 or idx >= nk:
            raise ValueError(f"k_index={k_index} is out of bounds for nk={nk}")
        k_indices = [idx]

    # ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # render each selected k-plane
    written: list[Path] = []
    for idx in k_indices:
        # extract one plane
        x_k = x[:, :, idx]
        y_k = y[:, :, idx]
        z_k = z[:, :, idx]
        f_k = data[:, :, idx]

        # clip out-of-range values if requested
        if clip_below:
            f_plot = np.clip(f_k, level_min, level_max)
        else:
            f_plot = f_k

        # construct a clean rectilinear mesh for contourf
        x_vec = x_k[:, 0]
        y_vec = y_k[0, :]
        x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing="ij")

        # derive beta token from median value on this plane
        beta_value = float(np.median(z_k))
        beta_token = f"{int(round(beta_value)):04d}"
        out_file = out_dir / f"{prefix}_{beta_token}.png"

        # build and save figure
        fig, ax = plt.subplots(figsize=(8, 5))
        cf = ax.contourf(x_mesh, y_mesh, f_plot, levels=levels, cmap="coolwarm")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(field_name)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{field_name} at k={idx + 1} ({k_name}={beta_value:g})")
        ax.set_aspect("auto")
        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")

        # display only when requested (typically single-k workflows)
        if show:
            plt.show()
        else:
            plt.close(fig)

        written.append(out_file)

    # debug output for devs
    logger.debug(
        "rendered %d LST contour(s): field=%s, levels=[%g, %g], policy=%s",
        len(written),
        field_name,
        level_min,
        level_max,
        levels_policy,
    )

    return written
