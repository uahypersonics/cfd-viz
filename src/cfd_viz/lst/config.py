"""Typed configuration for LST plotting."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_LST_CONFIG_PATH = Path("cfd-viz-lst.toml")
DEFAULT_LST_INPUT = Path("growth_rate_with_nfact_amps.dat")


@dataclass(frozen=True, slots=True)
class LSTFieldConfig:
    """One standard LST contour field."""

    name: str
    prefix: str
    label: str
    levels_policy: str | None = None
    levels_count: int | None = None
    level_min: float | None = None
    level_max: float | None = None


@dataclass(frozen=True, slots=True)
class LSTPlotConfig:
    """Complete configuration for standard LST contour rendering."""

    input_path: Path = DEFAULT_LST_INPUT
    output_dir: Path = Path(".")
    dimensions: int = 3
    fields: tuple[LSTFieldConfig, ...] = field(
        default_factory=lambda: (
            LSTFieldConfig("-im(alpha)", "alpi_kc", r"$-\alpha_i$ [1/m]"),
            LSTFieldConfig("Nfac3", "nfac_kc", r"$N$"),
        )
    )
    xvar: str = "s"
    yvar: str = "freq,freq."
    kvar: str = "beta"
    x_label: str = r"$s$ [m]"
    y_label: str = r"$f$ [kHz]"
    x_scale: float = 1.0
    x_ticks: int = 6
    y_scale: float = 0.001
    y_ticks: int = 6
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_label: str = r"$k_c$"
    z_scale: float = 1.0
    z_min: float | None = None
    z_max: float | None = None
    all_k: bool = True
    k_index: int = 1
    levels_policy: str = "positive-rounded"
    levels_count: int = 60
    level_min: float | None = None
    level_max: float | None = None
    clip_below: bool = True
    draw_neutral_line: bool = True
    figure_width: float = 3.25
    figure_height: float = 2.5
    dpi: int = 300
    colormap: str = "jet"
    font_size: float = 11.0
    use_tex: bool = True
    show_beta: bool = True
    colorbar_shrink: float = 0.8


_DEFAULT_CONFIG_TEXT = r"""#--------------------------------------------------
# input
#--------------------------------------------------
[input]
# path to the LST Tecplot data file
path = "growth_rate_with_nfact_amps.dat"

# number of dimensions in the input data: 2 or 3
dimensions = 3

# horizontal coordinate variable in the input data
x_var = "s"

# vertical coordinate variable in the input data
y_var = "freq,freq."

# slicing coordinate variable for 3-D input
z_var = "beta"

#--------------------------------------------------
# output
#--------------------------------------------------
[output]
# directory for rendered PNG files
directory = "."

# PNG resolution in dots per inch
dpi = 300

#--------------------------------------------------
# contour customization
#--------------------------------------------------
[contour]
# clip values below the minimum contour level
clip_below = true

# draw the zero contour when the data crosses zero
draw_neutral_line = true

# growth-rate contour settings
[contour.alpi]
# variable name in the input data
variable = "-im(alpha)"

# displayed colorbar name
name = '$-\alpha_i$ [1/m]'

# output filename prefix
output_prefix = "alpi_kc"

# contour level policy: global-auto or positive-rounded
levels_policy = "positive-rounded"

# number of filled-contour levels
levels_count = 60

# explicit contour limits; uncomment both to override the level policy
# level_min = 0.0
# level_max = 7500.0

# N-factor contour settings
[contour.nfac]
# variable name in the input data
variable = "Nfac3"

# displayed colorbar name
name = '$N$'

# output filename prefix
output_prefix = "nfac_kc"

# contour level policy: global-auto or positive-rounded
levels_policy = "positive-rounded"

# number of filled-contour levels
levels_count = 60

# explicit contour limits; uncomment both to override the level policy
# level_min = 0.0
# level_max = 5.0

#--------------------------------------------------
# x axis
#--------------------------------------------------
[x_axis]
# displayed axis label
label = '$s$ [m]'

# multiplier from input units to displayed units
scale = 1.0

# approximate number of major ticks
ticks = 6

# displayed axis limits; uncomment both to set fixed limits
# minimum = 0.0
# maximum = 1.0

#--------------------------------------------------
# y axis
#--------------------------------------------------
[y_axis]
# displayed axis label
label = '$f$ [kHz]'

# multiplier from input units to displayed units
scale = 0.001

# approximate number of major ticks
ticks = 6

# displayed axis limits; uncomment both to set fixed limits
# minimum = 0.0
# maximum = 500.0

#--------------------------------------------------
# z axis
#--------------------------------------------------
[z_axis]
# displayed z-plane label for 3-D input
label = '$k_c$'

# multiplier from input units to displayed units
scale = 1.0

# render every z-plane when true
all_planes = true

# 1-based z-plane index used when all_planes is false
index = 1

# show the selected z-plane value on each contour
show_label = true

# displayed z limits used to select planes; uncomment both to filter
# minimum = 0.0
# maximum = 1000.0

#--------------------------------------------------
# plot style
#--------------------------------------------------
[style]
# figure dimensions in inches
figure_width = 3.25
figure_height = 2.5

# Matplotlib colormap
colormap = "jet"

# base font size in points
font_size = 11.0

# use LaTeX when a LaTeX executable is available
use_tex = true

# colorbar height relative to the contour axes
colorbar_shrink = 0.8

"""


def default_lst_config() -> LSTPlotConfig:
    """Return a fresh LST plotting configuration with built-in defaults."""

    config = LSTPlotConfig()
    return config


def load_lst_config(path: str | Path = DEFAULT_LST_CONFIG_PATH) -> LSTPlotConfig:
    """Load and validate an LST plotting configuration.

    Args:
        path: TOML configuration path.

    Returns:
        Validated plotting configuration. Relative data paths are resolved from
        the configuration file's directory.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If a section or setting is invalid.
    """

    config_path = Path(path)
    with config_path.open("rb") as stream:
        source = tomllib.load(stream)

    _validate_root_keys(source)
    input_values = _read_table(source, "input")
    output_values = _read_table(source, "output")
    contour_values = _read_table(source, "contour")
    x_axis_values = _read_table(source, "x_axis")
    y_axis_values = _read_table(source, "y_axis")
    z_axis_values = _read_table(source, "z_axis")
    style_values = _read_table(source, "style")

    defaults = default_lst_config()
    fields = _read_contours(contour_values, defaults.fields)
    base_dir = config_path.resolve().parent

    input_path = _resolve_path(base_dir, input_values.get("path", defaults.input_path))
    output_dir = _resolve_path(base_dir, output_values.get("directory", defaults.output_dir))

    config = LSTPlotConfig(
        input_path=input_path,
        output_dir=output_dir,
        dimensions=int(input_values.get("dimensions", defaults.dimensions)),
        fields=fields,
        xvar=str(input_values.get("x_var", defaults.xvar)),
        yvar=str(input_values.get("y_var", defaults.yvar)),
        kvar=str(input_values.get("z_var", defaults.kvar)),
        x_label=str(x_axis_values.get("label", defaults.x_label)),
        y_label=str(y_axis_values.get("label", defaults.y_label)),
        x_scale=float(x_axis_values.get("scale", defaults.x_scale)),
        x_ticks=int(x_axis_values.get("ticks", defaults.x_ticks)),
        y_scale=float(y_axis_values.get("scale", defaults.y_scale)),
        y_ticks=int(y_axis_values.get("ticks", defaults.y_ticks)),
        x_min=_optional_float(x_axis_values.get("minimum")),
        x_max=_optional_float(x_axis_values.get("maximum")),
        y_min=_optional_float(y_axis_values.get("minimum")),
        y_max=_optional_float(y_axis_values.get("maximum")),
        z_label=str(z_axis_values.get("label", defaults.z_label)),
        z_scale=float(z_axis_values.get("scale", defaults.z_scale)),
        z_min=_optional_float(z_axis_values.get("minimum")),
        z_max=_optional_float(z_axis_values.get("maximum")),
        all_k=bool(z_axis_values.get("all_planes", defaults.all_k)),
        k_index=int(z_axis_values.get("index", defaults.k_index)),
        levels_policy=str(contour_values.get("levels_policy", defaults.levels_policy)),
        levels_count=int(contour_values.get("levels_count", defaults.levels_count)),
        level_min=_optional_float(contour_values.get("level_min")),
        level_max=_optional_float(contour_values.get("level_max")),
        clip_below=bool(contour_values.get("clip_below", defaults.clip_below)),
        draw_neutral_line=bool(contour_values.get("draw_neutral_line", defaults.draw_neutral_line)),
        figure_width=float(style_values.get("figure_width", defaults.figure_width)),
        figure_height=float(style_values.get("figure_height", defaults.figure_height)),
        dpi=int(output_values.get("dpi", defaults.dpi)),
        colormap=str(style_values.get("colormap", defaults.colormap)),
        font_size=float(style_values.get("font_size", defaults.font_size)),
        use_tex=bool(style_values.get("use_tex", defaults.use_tex)),
        show_beta=bool(z_axis_values.get("show_label", defaults.show_beta)),
        colorbar_shrink=float(style_values.get("colorbar_shrink", defaults.colorbar_shrink)),
    )
    _validate_config(config)

    return config


def write_default_lst_config(
    path: str | Path = DEFAULT_LST_CONFIG_PATH, *, force: bool = False
) -> Path:
    """Write a complete default LST plotting configuration.

    Args:
        path: Destination TOML path.
        force: Replace an existing file when True.

    Returns:
        Path to the generated configuration file.

    Raises:
        FileExistsError: If the destination exists and force is False.
    """

    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"configuration already exists: {output_path}; use --force to replace it"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_DEFAULT_CONFIG_TEXT, encoding="utf-8")

    return output_path


def validate_lst_config(config: LSTPlotConfig) -> None:
    """Validate a programmatically constructed LST plotting configuration.

    Args:
        config: Plotting configuration to validate.

    Raises:
        ValueError: If any setting is invalid.
    """

    _validate_config(config)


def _read_table(source: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one optional TOML table after validating its type."""

    value = source.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")

    return value


def _read_contours(
    contour_values: dict[str, Any],
    defaults: tuple[LSTFieldConfig, ...],
) -> tuple[LSTFieldConfig, ...]:
    """Build contour settings from named child tables."""

    fields = []
    for contour_name, field_values in contour_values.items():
        if not isinstance(field_values, dict):
            continue
        if "variable" not in field_values:
            raise ValueError(f"[contour.{contour_name}] is missing 'variable'")
        variable = str(field_values["variable"])
        fields.append(_build_field_config(variable, field_values))

    if not fields:
        fields = list(defaults)

    return tuple(fields)


def _build_field_config(name: str, values: dict[str, Any]) -> LSTFieldConfig:
    """Infer an output prefix and colorbar label from a contour variable."""

    normalized_name = name.lower().replace(".", "")
    if normalized_name in {"-im(alpha)", "imag(alpha)", "alpha_i", "alpha_imag"}:
        prefix = "alpi_kc"
        label = r"$-\alpha_i$ [1/m]"
    elif normalized_name in {"nfac", "nfac3"}:
        prefix = "nfac_kc"
        label = r"$N$"
    else:
        token = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
        prefix = f"{token or 'lst'}_kc"
        label = name

    field_config = LSTFieldConfig(
        name=name,
        prefix=str(values.get("output_prefix", prefix)),
        label=str(values.get("name", label)),
        levels_policy=_optional_str(values.get("levels_policy")),
        levels_count=_optional_int(values.get("levels_count")),
        level_min=_optional_float(values.get("level_min")),
        level_max=_optional_float(values.get("level_max")),
    )
    return field_config


def _resolve_path(base_dir: Path, value: object) -> Path:
    """Resolve a configured path relative to its configuration file."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path

    return path


def _optional_float(value: object) -> float | None:
    """Convert an optional TOML value to float."""

    result = None if value is None else float(value)
    return result


def _optional_int(value: object) -> int | None:
    """Convert an optional TOML value to int."""

    result = None if value is None else int(value)
    return result


def _optional_str(value: object) -> str | None:
    """Convert an optional TOML value to string."""

    result = None if value is None else str(value)
    return result


def _validate_root_keys(source: dict[str, Any]) -> None:
    """Reject unknown root sections that are likely configuration mistakes."""

    expected = {"input", "output", "contour", "x_axis", "y_axis", "z_axis", "style"}
    unknown = set(source) - expected
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown LST configuration section(s): {names}")


def _validate_config(config: LSTPlotConfig) -> None:
    """Validate plotting settings at the configuration boundary."""

    errors: list[str] = []
    if config.dimensions not in {2, 3}:
        errors.append("input.dimensions must be 2 or 3")
    if config.k_index < 1:
        errors.append("z_axis.index must be at least 1")
    if config.levels_policy not in {"global-auto", "positive-rounded"}:
        errors.append("contour.levels_policy must be global-auto or positive-rounded")
    if config.levels_count < 2:
        errors.append("contour.levels_count must be at least 2")
    if (config.level_min is None) != (config.level_max is None):
        errors.append("contour.level_min and level_max must be set together")
    elif (
        config.level_min is not None
        and config.level_max is not None
        and config.level_max <= config.level_min
    ):
        errors.append("contour.level_max must be greater than level_min")
    if config.figure_width <= 0.0 or config.figure_height <= 0.0:
        errors.append("style figure dimensions must be positive")
    if config.font_size <= 0.0:
        errors.append("style.font_size must be positive")
    if not 0.0 < config.colorbar_shrink <= 1.0:
        errors.append("style.colorbar_shrink must be greater than 0 and at most 1")
    if config.dpi < 72:
        errors.append("output.dpi must be at least 72")
    if config.x_ticks < 2:
        errors.append("x_axis.ticks must be at least 2")
    if config.y_ticks < 2:
        errors.append("y_axis.ticks must be at least 2")
    if config.x_scale == 0.0 or config.y_scale == 0.0 or config.z_scale == 0.0:
        errors.append("axis scale factors cannot be zero")
    for axis_name, lower, upper in (
        ("x", config.x_min, config.x_max),
        ("y", config.y_min, config.y_max),
        ("z", config.z_min, config.z_max),
    ):
        if (lower is None) != (upper is None):
            errors.append(f"{axis_name}_axis.minimum and maximum must be set together")
        elif lower is not None and upper <= lower:
            errors.append(f"{axis_name}_axis.maximum must be greater than minimum")
    if not config.fields:
        errors.append("at least one field must be configured")
    for field_config in config.fields:
        if field_config.levels_policy is not None and field_config.levels_policy not in {
            "global-auto",
            "positive-rounded",
        }:
            errors.append(
                f"field {field_config.name!r} levels_policy must be global-auto or positive-rounded"
            )
        if field_config.levels_count is not None and field_config.levels_count < 2:
            errors.append(f"field {field_config.name!r} levels_count must be at least 2")
        if (field_config.level_min is None) != (field_config.level_max is None):
            errors.append(
                f"field {field_config.name!r} level_min and level_max must be set together"
            )
        elif (
            field_config.level_min is not None and field_config.level_max <= field_config.level_min
        ):
            errors.append(f"field {field_config.name!r} level_max must be greater than level_min")

    if errors:
        raise ValueError("invalid LST configuration: " + "; ".join(errors))
