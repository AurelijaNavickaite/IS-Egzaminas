"""2 etapas: chronologinis train / validation / test skaidymas."""

from __future__ import annotations

import pandas as pd

from data_loader import month_change_indices

SEED = 42

# ~70 % ir ~85 % ribos, priklijuotos prie artimiausio mėnesio virsmo
# (žr. EKSfm dokumentą, 4.2 sk.).
TRAIN_END = 31459  # 69.58 %, pabaiga prieš 2-ojo ciklo „apr“
VAL_END = 39752    # 87.93 %, pabaiga prieš 2-ojo ciklo „jun“


def _nearest_change(target: int, changes: list[int]) -> int:
    return min(changes, key=lambda c: abs(c - target))


def chronological_indices(n: int, changes: list[int] | None = None) -> tuple[int, int]:
    """Grąžina (train_end, val_end) pagal ~70/15/15 ir mėnesio virsmus."""
    if changes is None:
        train_end, val_end = TRAIN_END, VAL_END
        train_end = min(train_end, n)
        val_end = min(val_end, n)
        return train_end, val_end
    t70 = int(0.70 * n)
    t85 = int(0.85 * n)
    train_end = _nearest_change(t70, changes)
    val_end = _nearest_change(t85, changes)
    if train_end >= val_end:
        val_end = min(n, train_end + max(1, int(0.15 * n)))
    return train_end, val_end


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pirmieji ~70 % — train, viduriniai ~15 % — val, paskutiniai ~15 % — test."""
    changes = month_change_indices(df)
    train_end, val_end = chronological_indices(len(df), changes)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    print(
        f"Chronologinis skaidymas: train={len(train)} ({len(train)/len(df):.1%}), "
        f"val={len(val)} ({len(val)/len(df):.1%}), "
        f"test={len(test)} ({len(test)/len(df):.1%})"
    )
    print(f"Ribos (eilutės): train_end={train_end}, val_end={val_end}")
    return train, val, test


def temporal_shift_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tas pats mokymas (ankstyviausios ~70 %).
    Arti = vidurys (~15 %), toli = vėliausios (~15 %).
    """
    changes = month_change_indices(df)
    train_end, val_end = chronological_indices(len(df), changes)
    train = df.iloc[:train_end].copy()
    near = df.iloc[train_end:val_end].copy()
    far = df.iloc[val_end:].copy()
    print(
        f"Laiko poslinkis: train={len(train)} (ankstyviausios), "
        f"arti={len(near)} (vidurys), toli={len(far)} (vėliausios)"
    )
    return train, near, far
