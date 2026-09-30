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
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from .config import DEFAULT_LST_INPUT, LSTPlotConfig, default_lst_config

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# publication rendering settings
# --------------------------------------------------
_DISPLAY_LABELS = {
    "s": r"$s$ [m]",
    "freq": r"$f$ [kHz]",
    "frequency": r"$f$ [kHz]",
    "-im(alpha)": r"$-\alpha_i$ [1/m]",
    "imag(alpha)": r"$-\alpha_i$ [1/m]",
    "alpha_i": r"$-\alpha_i$ [1/m]",
    "alpha_imag": r"$-\alpha_i$ [1/m]",
    "nfac": r"$N$",
    "nfac3": r"$N$",
}


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

    normalized_names = {name.lower(): name for name in flow}
    for name in candidates:
        normalized_name = name.lower()
        if normalized_name in normalized_names:
            return normalized_names[normalized_name]

    # include available keys in error for user debugging
    available = list(flow.keys())
    raise KeyError(
        f"none of the requested variables were found: {candidates}; available={available}"
    )


# --------------------------------------------------
# prepare display values and labels
# --------------------------------------------------
def _prepare_axis(
    values: np.ndarray,
    name: str,
    *,
    scale: float | None = None,
    label: str | None = None,
) -> tuple[np.ndarray, str]:
    """Scale an axis to readable units and return its publication label."""

    normalized_name = name.lower().replace(".", "")
    display_scale = scale
    if display_scale is None:
        display_scale = 0.001 if normalized_name in {"freq", "frequency"} else 1.0

    display_values = values * display_scale
    display_label = label if label is not None else _DISPLAY_LABELS.get(normalized_name, name)

    return display_values, display_label


# --------------------------------------------------
# resolve a publication label for a plotted field
# --------------------------------------------------
def _field_label(name: str) -> str:
    """Return a physical display label while preserving unknown field names."""

    normalized_name = name.lower().replace(".", "")
    return _DISPLAY_LABELS.get(normalized_name, name)


# --------------------------------------------------
# set axis bounds on labeled major ticks
# --------------------------------------------------
def _set_ticked_limits(
    ax: object,
    axis: str,
    values: np.ndarray,
    tick_count: int,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Set major ticks and limits so no unticked padding surrounds the data."""

    if lower is not None and upper is not None:
        ticks = np.linspace(lower, upper, tick_count)
    else:
        locator = MaxNLocator(nbins=tick_count, min_n_ticks=min(4, tick_count))
        ticks = locator.tick_values(float(np.nanmin(values)), float(np.nanmax(values)))
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xlim(float(ticks[0]), float(ticks[-1]))
    else:
        ax.set_yticks(ticks)
        ax.set_ylim(float(ticks[0]), float(ticks[-1]))


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
# build publication rendering settings
# --------------------------------------------------
def _build_plot_style(font_size: float, use_tex: bool) -> dict[str, object]:
    """Build Matplotlib settings from explicit presentation options."""

    latex_available = shutil.which("latex") is not None
    plot_style: dict[str, object] = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": use_tex and latex_available,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.8,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.facecolor": "white",
    }

    return plot_style


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
# public API: render standard LST contour products
# --------------------------------------------------
def render_standard_lst_contours(
    path: str | Path = DEFAULT_LST_INPUT,
    *,
    xvar: str = "s",
    yvar: str = "freq,freq.",
    kvar: str = "beta",
    out_dir: str | Path = ".",
    all_k: bool = True,
    k_index: int = 1,
    levels_policy: str = "positive-rounded",
    levels_count: int = 60,
    clip_below: bool = True,
    dpi: int = 300,
    show: bool = False,
) -> list[Path]:
    """Render the standard growth-rate and Nfac3 contour products.

    Args:
        path: LST data file. Defaults to ``growth_rate_with_nfact_amps.dat``.
        xvar: Candidate names for the x-axis variable.
        yvar: Candidate names for the y-axis variable.
        kvar: Candidate names for the beta variable.
        out_dir: Directory to write PNG images.
        all_k: When True, render all beta planes.
        k_index: 1-based beta index when ``all_k`` is False.
        levels_policy: Contour policy: ``global-auto`` or ``positive-rounded``.
        levels_count: Number of contour levels.
        clip_below: Clip values below the minimum contour level.
        dpi: PNG output resolution.
        show: Display figures interactively.

    Returns:
        Paths to all growth-rate and N-factor figures.
    """

    defaults = default_lst_config()
    config = replace(
        defaults,
        input_path=Path(path),
        output_dir=Path(out_dir),
        xvar=xvar,
        yvar=yvar,
        kvar=kvar,
        all_k=all_k,
        k_index=k_index,
        levels_policy=levels_policy,
        levels_count=levels_count,
        clip_below=clip_below,
        dpi=dpi,
    )
    written = render_configured_lst_contours(config, show=show)

    return written


# --------------------------------------------------
# public API: render an LST plotting configuration
# --------------------------------------------------
def render_configured_lst_contours(config: LSTPlotConfig, *, show: bool = False) -> list[Path]:
    """Render all fields selected by an LST plotting configuration.

    Args:
        config: Validated LST plotting configuration.
        show: Display figures interactively.

    Returns:
        Paths to all rendered contour figures.
    """

    written: list[Path] = []
    for field_config in config.fields:
        files = render_lst_contours(
            path=config.input_path,
            field=field_config.name,
            dimensions=config.dimensions,
            xvar=config.xvar,
            yvar=config.yvar,
            kvar=config.kvar,
            all_k=config.all_k,
            k_index=config.k_index,
            out_dir=config.output_dir,
            prefix=field_config.prefix,
            levels_policy=(
                field_config.levels_policy
                if field_config.levels_policy is not None
                else config.levels_policy
            ),
            levels_count=(
                field_config.levels_count
                if field_config.levels_count is not None
                else config.levels_count
            ),
            level_min_override=(
                field_config.level_min if field_config.level_min is not None else config.level_min
            ),
            level_max_override=(
                field_config.level_max if field_config.level_max is not None else config.level_max
            ),
            clip_below=config.clip_below,
            dpi=config.dpi,
            show=show,
            figure_size=(config.figure_width, config.figure_height),
            colormap=config.colormap,
            font_size=config.font_size,
            use_tex=config.use_tex,
            colorbar_shrink=config.colorbar_shrink,
            x_label=config.x_label,
            y_label=config.y_label,
            field_label=field_config.label,
            x_scale=config.x_scale,
            y_scale=config.y_scale,
            draw_neutral_line=config.draw_neutral_line,
            show_beta=config.show_beta,
            x_ticks=config.x_ticks,
            y_ticks=config.y_ticks,
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
            z_label=config.z_label,
            z_scale=config.z_scale,
            z_min=config.z_min,
            z_max=config.z_max,
        )
        written.extend(files)

    return written


# --------------------------------------------------
# public API: render a collection with shared contour bounds
# --------------------------------------------------
def render_configured_lst_collection(
    config: LSTPlotConfig,
    input_files: Sequence[str | Path],
    *,
    output_dir: str | Path | None = None,
    prefix_suffixes: Sequence[str | None] | None = None,
    single_plane: bool = False,
    show: bool = False,
) -> list[Path]:
    """Render multiple LST files with one contour range per configured field.

    Args:
        config: Validated LST plotting configuration.
        input_files: Ordered input files to render.
        output_dir: Optional output-directory override.
        prefix_suffixes: Optional filename suffix corresponding to each input.
        single_plane: Render only the first z-plane from each input.
        show: Display figures interactively.

    Returns:
        Paths to all rendered contour figures.

    Raises:
        ValueError: If no inputs are provided or suffix counts do not match.
        FileNotFoundError: If an input file does not exist.
    """

    # normalize and validate collection inputs
    paths = [Path(path) for path in input_files]
    if not paths:
        raise ValueError("at least one LST input file is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")

    suffixes = list(prefix_suffixes) if prefix_suffixes is not None else [None] * len(paths)
    if len(suffixes) != len(paths):
        raise ValueError("prefix_suffixes must match the number of input files")

    # compute shared bounds only when comparing multiple files
    render_config = config
    if len(paths) > 1:
        render_config = _apply_shared_contour_bounds(config, paths)

    # render each file through the standard configured renderer
    written: list[Path] = []
    selected_output_dir = Path(output_dir) if output_dir is not None else config.output_dir
    for path, suffix in zip(paths, suffixes, strict=True):
        fields = render_config.fields
        if suffix is not None:
            fields = tuple(
                replace(field, prefix=f"{field.prefix}_{suffix}") for field in fields
            )

        file_config = replace(
            render_config,
            input_path=path,
            output_dir=selected_output_dir,
            fields=fields,
            all_k=False if single_plane else render_config.all_k,
            k_index=1 if single_plane else render_config.k_index,
        )
        files = render_configured_lst_contours(file_config, show=show)
        written.extend(files)

    return written


# --------------------------------------------------
# compute shared bounds for a configured collection
# --------------------------------------------------
def _apply_shared_contour_bounds(
    config: LSTPlotConfig, input_files: Sequence[Path]
) -> LSTPlotConfig:
    """Return a config with one derived range per field across all inputs."""

    from cfd_io import read_file

    fields = []
    for field in config.fields:
        if field.level_min is not None and field.level_max is not None:
            fields.append(field)
            continue

        global_min: float | None = None
        global_max: float | None = None
        for path in input_files:
            dataset = read_file(str(path))
            field_name = _pick_flow_var(dataset.flow, field.name)
            values = dataset.flow[field_name].data
            local_min = float(np.nanmin(values))
            local_max = float(np.nanmax(values))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

        if global_min is None or global_max is None:
            raise ValueError(f"could not compute contour bounds for {field.name!r}")

        levels_policy = field.levels_policy or config.levels_policy
        levels = _build_levels(
            np.asarray([global_min, global_max]),
            policy=levels_policy,
            count=field.levels_count or config.levels_count,
        )
        fields.append(
            replace(
                field,
                level_min=float(levels[0]),
                level_max=float(levels[-1]),
            )
        )

    configured = replace(config, fields=tuple(fields))
    return configured


# --------------------------------------------------
# public API: render one or many LST contour plots
# --------------------------------------------------
def render_lst_contours(
    *,
    path: str | Path,
    field: str = "-im(alpha)",
    dimensions: int = 3,
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
    figure_size: tuple[float, float] = (3.25, 2.5),
    colormap: str = "jet",
    font_size: float = 11.0,
    use_tex: bool = True,
    colorbar_shrink: float = 0.8,
    x_label: str | None = None,
    y_label: str | None = None,
    field_label: str | None = None,
    x_scale: float | None = None,
    y_scale: float | None = None,
    draw_neutral_line: bool = True,
    show_beta: bool = True,
    x_ticks: int = 6,
    y_ticks: int = 6,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_label: str = r"$k_c$",
    z_scale: float = 1.0,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[Path]:
    """Render LST contours from a flow-only table dataset.

    Args:
        path: Input file path readable by cfd-io.
        field: Flow variable name for contour values.
        dimensions: Input dimensionality: 2 for one contour or 3 for z-plane slices.
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
        figure_size: Figure width and height in inches.
        colormap: Matplotlib colormap name.
        font_size: Base figure font size in points.
        use_tex: Use LaTeX text rendering when a LaTeX executable is available.
        colorbar_shrink: Colorbar height relative to the contour axes.
        x_label: Optional x-axis display label.
        y_label: Optional y-axis display label.
        field_label: Optional colorbar display label.
        x_scale: Optional multiplier applied to x-axis coordinates.
        y_scale: Optional multiplier applied to y-axis coordinates.
        draw_neutral_line: Draw the zero contour when the field crosses zero.
        show_beta: Annotate each 3-D slice with its z-plane value.
        x_ticks: Approximate number of x-axis major ticks.
        y_ticks: Approximate number of y-axis major ticks.
        x_min: Optional displayed x-axis minimum.
        x_max: Optional displayed x-axis maximum.
        y_min: Optional displayed y-axis minimum.
        y_max: Optional displayed y-axis maximum.
        z_label: Label used for the selected z-plane annotation.
        z_scale: Multiplier from input z units to displayed units.
        z_min: Optional displayed lower z limit used to select planes.
        z_max: Optional displayed upper z limit used to select planes.

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
    field_name = _pick_flow_var(flow, field)

    # extract mapped arrays
    x = flow[x_name].data
    y = flow[y_name].data
    data = flow[field_name].data

    # validate shape compatibility
    if x.shape != y.shape or x.shape != data.shape:
        raise ValueError(
            f"mapped variables must have matching shapes: "
            f"x={x.shape}, y={y.shape}, field={data.shape}"
        )
    if dimensions == 2:
        if data.ndim == 3 and data.shape[2] == 1:
            x = x[:, :, 0]
            y = y[:, :, 0]
            data = data[:, :, 0]
        elif data.ndim != 2:
            raise ValueError(f"expected 2-D data, got shape={data.shape}")
        z = None
    elif dimensions == 3:
        k_name = _pick_flow_var(flow, kvar)
        z = flow[k_name].data
        if x.shape != z.shape:
            raise ValueError(
                "mapped variables must have matching shapes: "
                f"x={x.shape}, y={y.shape}, z={z.shape}, field={data.shape}"
            )
        if data.ndim != 3:
            raise ValueError(f"expected 3-D data for z-slices, got shape={data.shape}")
    else:
        raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")

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

    # decide which z-planes to render
    if dimensions == 2:
        k_indices: list[int | None] = [None]
    else:
        nk = data.shape[2]
        if all_k:
            k_indices = list(range(nk))
            if z_min is not None and z_max is not None:
                k_indices = [
                    idx
                    for idx in k_indices
                    if z_min <= float(np.median(z[:, :, idx])) * z_scale <= z_max
                ]
        else:
            idx = k_index - 1
            if idx < 0 or idx >= nk:
                raise ValueError(f"z_axis.index={k_index} is out of bounds for nz={nk}")
            k_indices = [idx]

    # ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # render each selected k-plane
    written: list[Path] = []
    for idx in k_indices:
        # extract one plane
        if idx is None:
            x_k = x
            y_k = y
            f_k = data
            z_value = None
            out_file = out_dir / f"{prefix}.png"
        else:
            x_k = x[:, :, idx]
            y_k = y[:, :, idx]
            f_k = data[:, :, idx]
            z_value = float(np.median(z[:, :, idx])) * z_scale
            z_token = f"{int(round(float(np.median(z[:, :, idx])))):04d}"
            out_file = out_dir / f"{prefix}_{z_token}.png"

        # clip out-of-range values if requested
        if clip_below:
            f_plot = np.clip(f_k, level_min, level_max)
        else:
            f_plot = f_k

        # construct a clean rectilinear mesh for contourf
        x_vec = x_k[:, 0]
        y_vec = y_k[0, :]
        x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing="ij")
        x_plot, display_x_label = _prepare_axis(x_mesh, x_name, scale=x_scale, label=x_label)
        y_plot, display_y_label = _prepare_axis(y_mesh, y_name, scale=y_scale, label=y_label)
        display_field_label = field_label if field_label is not None else _field_label(field_name)

        # build and save figure with a consistent publication style
        plot_style = _build_plot_style(font_size, use_tex)
        with plt.rc_context(plot_style):
            fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
            contour = ax.contourf(
                x_plot,
                y_plot,
                f_plot,
                levels=levels,
                cmap=colormap,
                extend="max",
            )

            # mark the neutral-stability boundary when the source field crosses zero
            if draw_neutral_line and float(np.nanmin(f_k)) < 0.0 < float(np.nanmax(f_k)):
                ax.contour(
                    x_plot,
                    y_plot,
                    f_k,
                    levels=[0.0],
                    colors="white",
                    linewidths=0.9,
                )

            # label the field and selected spanwise wavenumber
            colorbar = fig.colorbar(
                contour,
                ax=ax,
                pad=0.025,
                aspect=28,
                shrink=colorbar_shrink,
            )
            colorbar.set_label(display_field_label)
            colorbar.set_ticks([level_min, level_max])
            colorbar.outline.set_linewidth(0.7)
            colorbar.ax.tick_params(direction="out", length=3, width=0.7)

            ax.set_xlabel(display_x_label)
            ax.set_ylabel(display_y_label)
            if show_beta and z_value is not None:
                ax.set_title(
                    f"{z_label} = {z_value:g}",
                    fontsize=font_size,
                    pad=8.0,
                )
            ax.set_aspect("auto")
            _set_ticked_limits(ax, "x", x_plot, x_ticks, x_min, x_max)
            _set_ticked_limits(ax, "y", y_plot, y_ticks, y_min, y_max)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.6)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
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
