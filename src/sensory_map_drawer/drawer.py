"""Interactive tool to draw sensory maps on a grid over a body image and save them as CSV.

Python replacement for the MATLAB mapping_sensation_gui.m (same grid).

    draw-map maps.csv --patient P01                          # default hand image, 40 x 27 grid
    draw-map maps.csv --patient P01 --session T0             # several time points per patient
    draw-map maps.csv --patient P01 --maps lost referred     # map names of a new CSV
    draw-map maps.csv --patient X01 --image foot.jpg --rows 60

Mouse: left click / drag = paint (on an empty cell) or erase (on a filled cell),
right click / drag = erase.
Keys:  1-9 select map, z undo, o show/hide other maps, s save.
If the window is closed with unsaved changes they are written to an .npz file
(see console; default: next to the CSV).
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button, RadioButtons

from . import maps as sensory_maps
from .maps import grid_shape

DEFAULT_MAPS = ["lost", "median_restored", "ulnar_restored"]
# Colour of a map whose name starts with the key; other maps get the next of _OTHERS.
KNOWN_COLORS = {"lost": (0.15, 0.15, 0.15), "median": (0.392, 0.208, 0.514), "ulnar": (0.263, 0.647, 0.812)}
_OTHERS = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def map_colors(names, known=None):
    known = KNOWN_COLORS if known is None else known
    colors, spare = {}, iter(_OTHERS)
    for n in names:
        c = next((c for k, c in known.items() if n.startswith(k)), None)
        colors[n] = to_rgb(c or next(spare))
    return colors


def _safe(text: str) -> str:
    return text.replace(" ", "_").replace("/", "-")


class MapDrawer:
    """The drawing window. ``on_save(masks)`` is called on save.

    hand_area: cell count of the whole body part, to show the map as % of it.
    cache_dir: where unsaved changes go if the window is closed (default: cwd).
    colors: {name prefix: colour}, overrides KNOWN_COLORS.
    """

    def __init__(self, image, masks: dict[str, np.ndarray], title: str, on_save: Callable,
                 hand_area: int | None = None, cache_dir: Path | None = None, colors: dict | None = None):
        self.image = image
        self.masks = {k: v.astype(bool).copy() for k, v in masks.items()}
        self.names = list(masks)
        self.colors = map_colors(self.names, colors)
        self.rows, self.cols = next(iter(masks.values())).shape
        _, _, self.ch, self.cw = grid_shape(image, self.rows)
        self.title = title
        self.on_save = on_save
        self.hand_area = hand_area
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd()
        self.active = self.names[0]
        self.show_others = True
        self.undo_stack = []
        self.dirty = False
        self._paint = None  # value applied while dragging, None = not dragging

        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_axes([0.02, 0.04, 0.62, 0.9])
        self.ax.imshow(image, interpolation="bilinear")
        extent = (0, self.cols * self.cw, self.rows * self.ch, 0)
        self.overlay = self.ax.imshow(
            np.zeros((self.rows, self.cols, 4)), extent=extent, interpolation="nearest"
        )
        for r in range(self.rows + 1):
            self.ax.axhline(r * self.ch, color="0.6", lw=0.3, alpha=0.5)
        for c in range(self.cols + 1):
            self.ax.axvline(c * self.cw, color="0.6", lw=0.3, alpha=0.5)
        self.ax.set_xlim(0, image.shape[1])
        self.ax.set_ylim(image.shape[0], 0)
        self.ax.set_axis_off()

        rax = self.fig.add_axes([0.67, 0.55, 0.31, 0.04 * len(self.names) + 0.04])
        self.radio = RadioButtons(rax, self.names)
        for label, n in zip(self.radio.labels, self.names):
            label.set_color(self.colors[n])
        self.radio.on_clicked(self.select)

        self.buttons = []
        for i, (text, cb) in enumerate(
            [("Save (s)", self.save), ("Undo (z)", self.undo), ("Clear map", self.clear),
             ("Others on/off (o)", self.toggle_others)]
        ):
            bax = self.fig.add_axes([0.67, 0.42 - i * 0.07, 0.31, 0.055])
            b = Button(bax, text)
            b.on_clicked(lambda _e, cb=cb: cb())
            self.buttons.append(b)
        self.fig.text(
            0.67, 0.06,
            "Left drag: paint / erase\nRight drag: erase\nKeys 1-9: select map",
            fontsize=8, va="bottom",
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.redraw()

    # --- model ---------------------------------------------------------------
    def cell(self, x, y):
        if x is None or y is None:
            return None
        r, c = int(y // self.ch), int(x // self.cw)
        return (r, c) if 0 <= r < self.rows and 0 <= c < self.cols else None

    def set_cell(self, rc, value):
        m = self.masks[self.active]
        if m[rc] != value:
            m[rc] = value
            self.dirty = True
            self.redraw()

    def select(self, name):
        self.active = name
        self.redraw()

    def undo(self):
        if self.undo_stack:
            name, mask = self.undo_stack.pop()
            self.masks[name] = mask
            self.dirty = True
            self.redraw()

    def clear(self):
        self.undo_stack.append((self.active, self.masks[self.active].copy()))
        self.masks[self.active][:] = False
        self.dirty = True
        self.redraw()

    def toggle_others(self):
        self.show_others = not self.show_others
        self.redraw()

    def save(self):
        self.on_save(self.masks)
        self.dirty = False
        self.redraw()

    # --- events --------------------------------------------------------------
    def on_press(self, event):
        if event.inaxes is not self.ax:
            return
        rc = self.cell(event.xdata, event.ydata)
        if rc is None:
            return
        self.undo_stack.append((self.active, self.masks[self.active].copy()))
        self._paint = False if event.button == 3 else not self.masks[self.active][rc]
        self.set_cell(rc, self._paint)

    def on_motion(self, event):
        if self._paint is None or event.inaxes is not self.ax:
            return
        rc = self.cell(event.xdata, event.ydata)
        if rc is not None:
            self.set_cell(rc, self._paint)

    def on_release(self, _event):
        self._paint = None

    def on_key(self, event):
        if event.key and event.key.isdigit() and 1 <= int(event.key) <= len(self.names):
            self.radio.set_active(int(event.key) - 1)
        elif event.key == "z":
            self.undo()
        elif event.key == "o":
            self.toggle_others()
        elif event.key == "s":
            self.save()

    def on_close(self, _event):
        if self.dirty:
            out = self.cache_dir / f"drawmap_unsaved_{_safe(self.title)}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **self.masks)
            print(f"Unsaved changes written to {out} (load with np.load)")

    # --- view ----------------------------------------------------------------
    def redraw(self):
        rgba = np.zeros((self.rows, self.cols, 4))
        layers = [n for n in self.names if n != self.active] if self.show_others else []
        for n, alpha in [(n, 0.15) for n in layers] + [(self.active, 0.55)]:
            m = self.masks[n]
            rgba[m, :3] = self.colors[n]
            rgba[m, 3] = alpha
        self.overlay.set_data(rgba)
        n = int(self.masks[self.active].sum())
        pct = f" ({n / self.hand_area * 100:.1f}% of hand)" if self.hand_area else ""
        star = " *unsaved*" if self.dirty else ""
        self.ax.set_title(f"{self.title} | {self.active}: {n} cells{pct}{star}", fontsize=9)
        self.fig.canvas.draw_idle()


def add_arguments(ap: argparse.ArgumentParser, csv_required: bool = True) -> None:
    """The command line options of draw-map (for wrappers that add their own)."""
    if csv_required:
        ap.add_argument("csv", type=Path, help="CSV to read/write (created on first save)")
    ap.add_argument("--patient", required=True, help="patient ID, e.g. P01")
    ap.add_argument("--session", help="time point, e.g. T0; for CSVs with several sessions per patient")
    ap.add_argument("--maps", nargs="+", help=f"map names for a new CSV (default: {' '.join(DEFAULT_MAPS)})")
    ap.add_argument("--image", type=Path, default=sensory_maps.HAND_IMAGE,
                    help="body image (default: the hand of mapping_sensation_gui.m)")
    ap.add_argument("--rows", type=int, default=sensory_maps.GRID[0], help="grid rows (cols follow the image)")
    ap.add_argument("--hand-area", type=int,
                    help="cells of the whole body part, to show %% (default: known for the default hand image)")
    ap.add_argument("--cache-dir", type=Path, help="where to dump unsaved changes on close (default: next to the CSV)")


def run(csv: Path, patient: str, session: str | None = None, maps: list[str] | None = None,
        image: Path = sensory_maps.HAND_IMAGE, rows: int = sensory_maps.GRID[0], hand_area: int | None = None,
        cache_dir: Path | None = None, title: str | None = None, colors: dict | None = None,
        show: bool = True) -> MapDrawer:
    """Open the drawer on ``csv`` for one patient (and session). Raises ValueError on bad input."""
    csv, image = Path(csv), Path(image)
    img = mpimg.imread(image)
    rows, cols, _, _ = grid_shape(img, rows)

    entry = f"{patient} {session}" if session else patient
    if csv.exists():
        existing = sensory_maps.read_csv(csv)
        names = existing.names
        if maps and maps != names:
            raise ValueError(f"{csv} already has maps {names}; --maps must match or be omitted")
        if next(iter(existing.masks.values())).shape[1:] != (rows, cols):
            raise ValueError(f"{csv} has a different grid than {image.name} with {rows} rows ({rows} x {cols})")
        has_sessions = existing.sessions is not None
        if has_sessions != bool(session):
            raise ValueError(f"{csv} " + ("has sessions: give --session" if has_sessions else "has no sessions: omit --session"))
        if has_sessions:
            done = [s for p, s in zip(existing.patients, existing.sessions) if p == patient]
            print(f"Sessions already saved for {patient}: {', '.join(done) or 'none'}")
            known = sorted(set(existing.sessions))
            if session not in known:
                print(f"Warning: new session label {session!r} (in file: {', '.join(known)})")
        try:
            masks = existing.entry(patient, session)
            print(f"Editing existing maps of {entry} from {csv}")
        except KeyError:
            masks = {n: np.zeros((rows, cols), bool) for n in names}
    else:
        names = maps or DEFAULT_MAPS
        masks = {n: np.zeros((rows, cols), bool) for n in names}
        kind = "with" if session else "without"
        print(f"{csv} does not exist yet; it will be created on save with maps {names}, {kind} sessions")

    def on_save(m):
        sensory_maps.write_patient(csv, patient, m, session)
        print(f"Saved {entry} to {csv}")

    if hand_area is None and image.resolve() == sensory_maps.HAND_IMAGE.resolve() and rows == sensory_maps.GRID[0]:
        hand_area = sensory_maps.TOTAL_HAND_AREA
    drawer = MapDrawer(img, masks, title or f"{csv.stem} {entry}", on_save, hand_area,
                       cache_dir or csv.parent, colors)
    if show:
        plt.show()
    return drawer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)
    try:
        run(args.csv, args.patient, args.session, args.maps, args.image, args.rows, args.hand_area, args.cache_dir)
    except ValueError as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
