# sensory-map-drawer

Draw sensory maps (e.g. areas of lost or restored sensation) on a grid laid over a
body image, and save them as a tidy CSV. Python version of the MATLAB
`mapping_sensation_gui.m`: with the default hand image and 40 rows you get the same
40×27 grid of 15×15 px cells, so maps are comparable with the MATLAB ones.

## Install

Needs Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/), as a command available everywhere:

```
uv tool install git+https://github.com/annasparapani/sensory-map-drawer
```

Without uv: `pip install git+https://github.com/annasparapani/sensory-map-drawer`.

As a dependency of another uv project (gives you `draw-map` and the Python API there):

```
uv add git+https://github.com/annasparapani/sensory-map-drawer
```

Add `@v0.1.0` (or another tag) to the URL to pin a version. Update with
`uv tool upgrade sensory-map-drawer`.

For development, clone the repo and install it editable:

```
git clone https://github.com/annasparapani/sensory-map-drawer
uv tool install --editable ./sensory-map-drawer
```

## Draw

```
draw-map maps.csv --patient P01                          # default hand image, 40×27 grid
draw-map maps.csv --patient P01 --session T0             # several time points per patient
draw-map maps.csv --patient P01 --maps lost referred     # map names (the first save fixes them)
draw-map maps.csv --patient X01 --image foot.jpg --rows 60   # any other image / grid
```

- Left click / drag: paint on an empty cell, erase on a filled one. Right drag: erase.
- Keys: `1`–`9` select map, `z` undo, `o` show/hide the other maps, `s` save.
- The title shows the cell count (and % of the hand for the default image, or of
  `--hand-area` cells).
- Closing with unsaved changes writes them to `drawmap_unsaved_<...>.npz` next to the
  CSV (or in `--cache-dir`).

Saving replaces only that patient (and session) in the CSV; other entries are kept.
Map names, grid and the presence of a session column must match the existing file.

## CSV format

One row per patient × grid cell, `row`/`col` 0-based from the top-left of the image,
one 0/1 column per map:

```
patient,row,col,lost,median_restored,ulnar_restored
P01,0,0,0,0,0
```

With sessions: `patient,session,row,col,<maps>...`.

## Python API

```python
import sensory_map_drawer as smd

sm = smd.read_csv("maps.csv")              # SensoryMaps; merge=True unions each patient's sessions
sm.lost                                    # bool array (n_entries, rows, cols)
sm.entry("P01", "T0")                      # {map: (rows, cols)}
sm.session("T0"), sm.labels, sm.to_long()
smd.write_patient("maps.csv", "P02", {"lost": mask, ...})
smd.concat(sm_a, sm_b)                     # stack datasets with the same maps
smd.run("maps.csv", "P01", show=False)     # build the window without blocking (scripting / tests)
```

`MapDrawer(image, masks, title, on_save, ...)` is the window itself, for wrappers that
store maps elsewhere.

## Test

```
uv run pytest
```

The tests use `MPLBACKEND=Agg` and drive `MapDrawer` with fake mouse/key events.
