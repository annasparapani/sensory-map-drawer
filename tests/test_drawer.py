import os

os.environ.setdefault("MPLBACKEND", "Agg")

from types import SimpleNamespace

import numpy as np
import pytest

import sensory_map_drawer as smd


def click(d, r, c, button=1):
    x, y = (c + 0.5) * d.cw, (r + 0.5) * d.ch
    ev = SimpleNamespace(inaxes=d.ax, xdata=x, ydata=y, button=button)
    d.on_press(ev)
    return ev


def test_default_grid():
    import matplotlib.image as mpimg

    assert smd.grid_shape(mpimg.imread(smd.HAND_IMAGE), 40)[:2] == smd.GRID


def test_draw_save_reload(tmp_path):
    csv = tmp_path / "maps.csv"
    d = smd.run(csv, "P01", show=False)
    assert d.names == ["lost", "median_restored", "ulnar_restored"]
    assert d.hand_area == smd.TOTAL_HAND_AREA

    click(d, 10, 5)
    d.on_motion(SimpleNamespace(inaxes=d.ax, xdata=6.5 * d.cw, ydata=10.5 * d.ch))
    d.on_release(None)
    d.on_key(SimpleNamespace(key="2"))
    assert d.active == "median_restored"
    click(d, 20, 20); d.on_release(None)
    click(d, 20, 20, button=3); d.on_release(None)  # right click erases
    click(d, 21, 21); d.on_release(None)
    d.on_key(SimpleNamespace(key="z"))  # undo the last stroke
    assert not d.masks["median_restored"].any()
    click(d, 1, 1); d.on_release(None)
    d.on_key(SimpleNamespace(key="s"))
    assert not d.dirty

    sm = smd.read_csv(csv)
    assert sm.patients == ["P01"] and sm.sessions is None
    assert sm.lost.shape == (1, *smd.GRID)
    assert set(zip(*np.nonzero(sm.lost[0]))) == {(10, 5), (10, 6)}
    assert set(zip(*np.nonzero(sm.median_restored[0]))) == {(1, 1)}

    # reopening edits the saved maps, a second patient is added alongside
    assert smd.run(csv, "P01", show=False).masks["lost"].sum() == 2
    d2 = smd.run(csv, "P02", show=False)
    click(d2, 0, 0); d2.save()
    assert smd.read_csv(csv).patients == ["P01", "P02"]


def test_sessions_and_checks(tmp_path):
    csv = tmp_path / "maps.csv"
    for s, cell in [("T0", (3, 3)), ("T1", (4, 4))]:
        d = smd.run(csv, "P01", session=s, maps=["lost", "referred"], show=False)
        click(d, *cell); d.save()
    sm = smd.read_csv(csv)
    assert sm.labels == ["P01 T0", "P01 T1"]
    assert smd.read_csv(csv, merge=True).lost[0].sum() == 2
    with pytest.raises(ValueError, match="has sessions"):
        smd.run(csv, "P01", show=False)
    with pytest.raises(ValueError, match="already has maps"):
        smd.run(csv, "P01", session="T2", maps=["other"], show=False)
    with pytest.raises(ValueError, match="different grid"):
        smd.run(csv, "P01", session="T2", rows=60, show=False)


def test_unsaved_on_close(tmp_path):
    d = smd.run(tmp_path / "maps.csv", "P01", show=False)
    click(d, 2, 2)
    d.on_close(None)
    (out,) = tmp_path.glob("drawmap_unsaved_*.npz")
    assert np.load(out)["lost"][2, 2]
