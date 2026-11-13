"""Sensory maps stored as 0/1 masks on a grid laid over a body image.

CSV format, one row per patient x grid cell::

    patient,row,col,<map1>,<map2>,...
    S01,0,0,0,0,...

or, with several time points per patient, with a ``session`` column::

    patient,session,row,col,<map1>,...
    P01,T0,0,0,0,...

row/col are 0-based from the top-left of the image; each map column is 0/1.
Map names are free (e.g. lost, median_restored, ulnar_restored).

The grid is the one of the MATLAB mapping GUI (mapping_sensation_gui.m): the
image is split into ``rows`` rows x ceil(rows * width / height) columns. With the
default hand image (``HAND_IMAGE``) and 40 rows this gives the 40 x 27 grid of
15 x 15 px cells.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).parent / "assets"
HAND_IMAGE = ASSETS / "hand_gui_image.jpg"
GRID = (40, 27)  # grid of HAND_IMAGE with 40 rows
TOTAL_HAND_AREA = 281  # cells covered by the whole hand on HAND_IMAGE / GRID
CELL = ["row", "col"]


def grid_shape(image: np.ndarray, rows: int) -> tuple[int, int, int, int]:
    """(rows, cols, cell_h, cell_w) exactly as mapping_sensation_gui.m."""
    h, w = image.shape[:2]
    cols = math.ceil(rows * w / h)
    return rows, cols, h // rows, w // cols


def _keys(df: pd.DataFrame) -> list[str]:
    return ["patient", "session"] if "session" in df.columns else ["patient"]


@dataclass
class SensoryMaps:
    """Masks per map name, each of shape (n_entries, rows, cols), bool.

    One entry per patient, or per patient x session when ``sessions`` is set
    (then ``patients[i]`` and ``sessions[i]`` describe entry i).
    Maps are also available as attributes: ``sm.lost`` == ``sm.masks["lost"]``.
    ``missing[name]`` (bool, n_entries) flags entries where map ``name`` was not
    measured; their mask is all False. Maps absent from ``missing`` are measured
    for every entry.
    """

    patients: list[str]
    masks: dict[str, np.ndarray]
    sessions: list[str] | None = None
    missing: dict[str, np.ndarray] = field(default_factory=dict)

    def __getattr__(self, name):
        masks = self.__dict__.get("masks", {})
        if name in masks:
            return masks[name]
        raise AttributeError(name)

    @property
    def names(self) -> list[str]:
        return list(self.masks)

    def measured(self, name: str) -> np.ndarray:
        """Bool per entry: was map ``name`` measured?"""
        n = len(self.patients)
        return ~self.missing.get(name, np.zeros(n, bool))

    def mark_missing(self, name: str, idx=slice(None)) -> None:
        """Flag map ``name`` as not measured for entries ``idx`` (default all) and clear it."""
        miss = ~self.measured(name)
        miss[idx] = True
        self.missing[name] = miss
        self.masks[name][miss] = False

    @property
    def labels(self) -> list[str]:
        """Entry labels for plots: "P01" or "P01 T0"."""
        if self.sessions is None:
            return list(self.patients)
        return [f"{p} {s}" for p, s in zip(self.patients, self.sessions)]

    def entry(self, patient: str, session: str | None = None) -> dict[str, np.ndarray]:
        """Maps of one patient (and session). Raises KeyError if absent."""
        entries = list(zip(self.patients, self.sessions or [None] * len(self.patients)))
        try:
            i = entries.index((patient, session))
        except ValueError:
            raise KeyError((patient, session)) from None
        return {k: v[i] for k, v in self.masks.items()}

    def subset(self, idx) -> "SensoryMaps":
        """Entries at positions ``idx`` (list of ints)."""
        sessions = None if self.sessions is None else [self.sessions[i] for i in idx]
        return SensoryMaps([self.patients[i] for i in idx], {k: v[idx] for k, v in self.masks.items()}, sessions,
                           {k: v[idx] for k, v in self.missing.items()})

    def session(self, name: str) -> "SensoryMaps":
        """Only session ``name``, one entry per patient (no session labels)."""
        if self.sessions is None:
            raise ValueError("These maps have no sessions")
        sm = self.subset([i for i, s in enumerate(self.sessions) if s == name])
        sm.sessions = None
        return sm

    def to_long(self) -> pd.DataFrame:
        n, r, c = next(iter(self.masks.values())).shape
        e, row, col = np.meshgrid(np.arange(n), np.arange(r), np.arange(c), indexing="ij")
        e = e.ravel()
        df = pd.DataFrame({"patient": np.array(self.patients)[e]})
        if self.sessions is not None:
            df["session"] = np.array(self.sessions)[e]
        df["row"], df["col"] = row.ravel(), col.ravel()
        for name, m in self.masks.items():
            df[name] = m.reshape(-1).astype(int)
        return df

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "SensoryMaps":
        keys = _keys(df)
        i, entries = pd.MultiIndex.from_frame(df[keys]).factorize()  # in order of appearance
        shape = (len(entries), df["row"].max() + 1, df["col"].max() + 1)
        masks = {}
        for name in (c for c in df.columns if c not in keys + CELL):
            m = np.zeros(shape, bool)
            m[i, df["row"], df["col"]] = df[name].to_numpy(bool)
            masks[name] = m
        patients = [e[0] for e in entries]
        sessions = [e[1] for e in entries] if len(keys) > 1 else None
        return cls(patients, masks, sessions)


def read_long(csv: Path) -> pd.DataFrame:
    """The CSV as a DataFrame (patient/session kept as strings)."""
    return pd.read_csv(csv, dtype={"patient": str, "session": str})


def merge_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """One map per patient: a cell is 1 if it is 1 in any of the patient's sessions."""
    if "session" not in df.columns:
        return df
    maps = [c for c in df.columns if c not in ["patient", "session"] + CELL]
    return df.groupby(["patient"] + CELL, sort=False)[maps].max().reset_index()


def read_csv(csv: Path, merge: bool = False) -> SensoryMaps:
    """All entries in a sensory maps CSV; ``merge=True`` merges each patient's sessions."""
    df = read_long(csv)
    return SensoryMaps.from_long(merge_sessions(df) if merge else df)


def write_patient(csv: Path, patient: str, masks: dict[str, np.ndarray], session: str | None = None) -> None:
    """Insert or replace one patient's (or patient x session's) maps in ``csv``.

    The file is created if missing. Map names must match the columns already
    in the file, and ``session`` must be given if and only if the file has a
    session column.
    """
    csv = Path(csv)
    sessions = None if session is None else [session]
    new = SensoryMaps([patient], {k: v[None] for k, v in masks.items()}, sessions).to_long()
    if csv.exists():
        old = read_long(csv)
        if ("session" in old.columns) != (session is not None):
            need = "requires" if "session" in old.columns else "has no session column; do not give"
            raise ValueError(f"{csv} {need} a session")
        if list(old.columns) != list(new.columns):
            raise ValueError(f"Maps {list(masks)} do not match the columns of {csv}: {list(old.columns)}")
        same = old["patient"] == patient
        if session is not None:
            same &= old["session"] == session
        new = pd.concat([old[~same], new])
    csv.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(_keys(new) + CELL, kind="stable").to_csv(csv, index=False)


def concat(*sms: SensoryMaps) -> SensoryMaps:
    """Stack maps of several datasets (same map names and grid, no sessions)."""
    names = sms[0].names
    for sm in sms:
        if sm.names != names:
            raise ValueError(f"Map names differ: {sm.names} vs {names}")
        if sm.sessions is not None:
            raise ValueError("Select one session (sm.session(...)) or merge sessions before concat")
    patients = [p for sm in sms for p in sm.patients]
    if len(set(patients)) != len(patients):
        raise ValueError(f"Duplicate patient IDs across datasets: {patients}")
    missing = {k: np.concatenate([~sm.measured(k) for sm in sms]) for k in names if any(k in sm.missing for sm in sms)}
    return SensoryMaps(patients, {k: np.concatenate([sm.masks[k] for sm in sms]) for k in names}, missing=missing)
