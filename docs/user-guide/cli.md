# CLI Usage

## Mesh

Display a structured mesh:

```bash
cfd-viz mesh grid.su2
```

### Options

| Option | Short | Description |
|---|---|---|
| `--skip N` | `-s` | Plot every N-th grid line (default: 1) |
| `--output FILE` | `-o` | Save figure to file instead of showing |
| `--title TEXT` | `-t` | Plot title (default: "Mesh") |
| `--grid-in FILE` | `-g` | Separate grid file (for split formats) |

### Examples

```bash
# skip every 4th line for a dense mesh
cfd-viz mesh grid.su2 --skip 4

# save to file
cfd-viz mesh grid.su2 --skip 2 --output mesh.png

# split format
cfd-viz mesh grid.s8 --grid-in grid.s8
```

## Contour

Display a filled contour of a scalar field:

```bash
cfd-viz contour solution.h5 --field pres
```

### Options

| Option | Short | Description |
|---|---|---|
| `--field NAME` | `-f` | Flow variable to plot (required) |
| `--output FILE` | `-o` | Save figure to file instead of showing |
| `--title TEXT` | `-t` | Plot title (default: field name) |
| `--levels N` | `-l` | Number of contour levels (default: 50) |
| `--grid-in FILE` | `-g` | Separate grid file (for split formats) |

### Examples

```bash
# plot pressure contours
cfd-viz contour solution.h5 --field pres

# save with custom levels
cfd-viz contour solution.h5 --field temp --levels 100 --output temp.png

# split format with separate grid file
cfd-viz contour flow.s8 --grid-in grid.s8 --field pres
```

## Global Options

| Option | Short | Description |
|---|---|---|
| `--version` | `-V` | Show version and exit |
| `--debug` | | Enable debug logging |
| `--help` | | Show help and exit |
