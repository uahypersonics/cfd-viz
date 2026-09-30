"""LST visualization command group."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

# --------------------------------------------------
# LST command group
# --------------------------------------------------
lst_app = typer.Typer(
    help="Visualize LST table-like datasets.",
    no_args_is_help=True,
)


# --------------------------------------------------
# derive an output prefix from an LST field name
# --------------------------------------------------
def _lst_field_prefix(field: str) -> str:
    """Return a compact output prefix for a requested LST field."""

    candidates = [name.strip().lower() for name in field.split(",")]
    alpha_names = {"-im(alpha)", "imag(alpha)", "alpha_i", "alpha_imag"}
    if any(name in alpha_names for name in candidates):
        prefix = "alpi_kc"
    elif any(name in {"nfac", "nfac3"} for name in candidates):
        prefix = "nfac_kc"
    else:
        field_token = re.sub(r"[^a-z0-9]+", "_", candidates[0]).strip("_")
        prefix = f"{field_token}_kc"

    return prefix


# --------------------------------------------------
# load an explicit or discovered LST configuration
# --------------------------------------------------
def _load_lst_command_config(config_path: Path | None):
    """Load an explicit or local config, otherwise return built-in defaults."""

    from cfd_viz.lst import DEFAULT_LST_CONFIG_PATH, default_lst_config, load_lst_config

    if config_path is not None:
        config = load_lst_config(config_path)
    elif DEFAULT_LST_CONFIG_PATH.exists():
        config = load_lst_config(DEFAULT_LST_CONFIG_PATH)
    else:
        config = default_lst_config()

    return config


# --------------------------------------------------
# LST init command
# --------------------------------------------------
@lst_app.command("init")
def cmd_lst_init(
    output: Annotated[
        Path,
        typer.Argument(help="LST plotting configuration to create."),
    ] = Path("cfd-viz-lst.toml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Replace an existing configuration."),
    ] = False,
) -> None:
    """Create an editable LST plotting configuration."""

    from cfd_viz.lst import write_default_lst_config

    try:
        written = write_default_lst_config(output, force=force)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"wrote {written}")


# --------------------------------------------------
# LST contours command
# --------------------------------------------------
@lst_app.command("contours")
def cmd_lst_contours(
    path: Annotated[
        Path | None,
        typer.Argument(help="LST data file (overrides config)."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="LST plotting TOML; defaults to cfd-viz-lst.toml when present.",
        ),
    ] = None,
    field: Annotated[
        str | None,
        typer.Option(
            "--field",
            "-f",
            help="Render one field instead of the standard -im(alpha) and Nfac3 plots.",
        ),
    ] = None,
    xvar: Annotated[
        str | None,
        typer.Option("--xvar", help="X-axis variable name or comma-separated aliases."),
    ] = None,
    yvar: Annotated[
        str | None,
        typer.Option("--yvar", help="Y-axis variable name or comma-separated aliases."),
    ] = None,
    kvar: Annotated[
        str | None,
        typer.Option("--kvar", help="K-sweep variable name or comma-separated aliases."),
    ] = None,
    all_k: Annotated[
        bool | None,
        typer.Option("--all-k/--single-k", help="Render all k-planes or one selected plane."),
    ] = None,
    k_index: Annotated[
        int | None,
        typer.Option("--k-index", "-k", min=1, help="1-based index used with --single-k."),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", "-o", help="Directory for output PNG files."),
    ] = None,
    prefix: Annotated[
        str | None,
        typer.Option("--prefix", "-p", help="Output prefix for a custom --field."),
    ] = None,
    levels_policy: Annotated[
        str | None,
        typer.Option(
            "--levels-policy",
            help="Contour levels policy: global-auto or positive-rounded.",
        ),
    ] = None,
    levels_count: Annotated[
        int | None,
        typer.Option("--levels-count", min=2, help="Number of contour levels."),
    ] = None,
    clip_below: Annotated[
        bool | None,
        typer.Option(
            "--clip-below/--no-clip-below",
            help="Clip values below the minimum contour level.",
        ),
    ] = None,
    dpi: Annotated[
        int | None,
        typer.Option("--dpi", min=72, help="PNG output DPI."),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show", help="Display plot window; single-k is recommended."),
    ] = False,
) -> None:
    """Render standard growth-rate and N-factor contours from LST data."""

    from cfd_viz.lst import (
        LSTFieldConfig,
        render_configured_lst_contours,
        validate_lst_config,
    )

    try:
        plot_config = _load_lst_command_config(config_path)
        updates: dict[str, object] = {}
        for name, value in (
            ("input_path", path),
            ("xvar", xvar),
            ("yvar", yvar),
            ("kvar", kvar),
            ("all_k", all_k),
            ("k_index", k_index),
            ("output_dir", out_dir),
            ("levels_policy", levels_policy),
            ("levels_count", levels_count),
            ("clip_below", clip_below),
            ("dpi", dpi),
        ):
            if value is not None:
                updates[name] = value

        if field is not None:
            output_prefix = prefix if prefix is not None else _lst_field_prefix(field)
            field_label = field
            if output_prefix == "alpi_kc":
                field_label = r"$-\alpha_i$ [1/m]"
            elif output_prefix == "nfac_kc":
                field_label = r"$N$"
            updates["fields"] = (LSTFieldConfig(field, output_prefix, field_label),)
        elif prefix is not None:
            raise ValueError("--prefix requires --field")

        plot_config = replace(plot_config, **updates)
        validate_lst_config(plot_config)

        # keep batch mode render-first to avoid opening many windows
        if show and plot_config.all_k:
            typer.echo(
                "warning: --show ignored with --all-k; files will be rendered only",
                err=True,
            )
            show = False

        files = render_configured_lst_contours(plot_config, show=show)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not files:
        typer.echo("no plots generated", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"wrote {len(files)} plot(s)")
    typer.echo(f"first: {files[0]}")
    typer.echo(f"last:  {files[-1]}")
