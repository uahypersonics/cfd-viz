# Python API

The plotting functions can be used directly in scripts and notebooks.

## Mesh

```python
from cfd_io import read_file
from cfd_viz.mesh import plot_mesh

ds = read_file("grid.su2")
plot_mesh(ds.grid.x, ds.grid.y, skip=2)
```

Save to file without showing:

```python
plot_mesh(ds.grid.x, ds.grid.y, skip=2, filename="mesh.png", show=False)
```

## Contour

```python
from cfd_io import read_file
from cfd_viz.contour import plot_contour

ds = read_file("solution.h5")
plot_contour(ds.grid.x, ds.grid.y, ds.flow["pres"].data, field_name="pres")
```

Save to file without showing:

```python
plot_contour(
    ds.grid.x, ds.grid.y, ds.flow["pres"].data,
    field_name="pres",
    levels=100,
    filename="pres.png",
    show=False,
)
```

## Working with NumPy Arrays Directly

The plotting functions accept plain NumPy arrays — no cfd-io dependency
required:

```python
import numpy as np
from cfd_viz.mesh import plot_mesh

x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 0.5, 30), indexing="ij")
plot_mesh(x, y)
```
