"""2 etapas: požymių paruošimas (be duomenų nutekėjimo iš val/test į train)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

CAT_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]
NUM_COLS_BASE = ["age", "balance", "day", "campaign", "pdays", "previous", "never_contacted"]


def _one_hot_encoder() -> OneHotEncoder:
    """Colab kartais turi senesnį sklearn (sparse=), naujesnis naudoja sparse_output=."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    pdays=-1 reiškia, kad klientas niekada nebuvo kontaktuotas anksčiau.
    Todėl kuriame never_contacted ir pdays=-1 pakeičiame 0, kad modelis
    nelaikytų -1 „neigiama trukme“.
    """
    out = df.copy()
    out["never_contacted"] = (out["pdays"] == -1).astype(int)
    out.loc[out["pdays"] == -1, "pdays"] = 0
    out["y_bin"] = (out["y"].astype(str).str.lower() == "yes").astype(int)
    return out


def cap_previous_from_train(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """previous maks. 275 iškraipo mokymąsi — apkerpame ties train 99-uoju procentiliu."""
    cap = float(train["previous"].quantile(0.99))
    print(f"previous apkirpimas (99-asis train procentilis) = {cap:.2f}")
    frames = []
    for part in (train, *others):
        p = part.copy()
        p["previous"] = p["previous"].clip(upper=cap)
        frames.append(p)
    return tuple(frames)


def numeric_cols(include_duration: bool) -> list[str]:
    cols = list(NUM_COLS_BASE)
    if include_duration:
        cols = cols + ["duration"]
    return cols


def build_sklearn_preprocessor(include_duration: bool) -> ColumnTransformer:
    """One-hot kategorijoms (unknown lieka atskira kategorija) + mastelio suvienodinimas skaičiams."""
    num = numeric_cols(include_duration)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            (
                "cat",
                _one_hot_encoder(),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def xy_frames(df: pd.DataFrame, include_duration: bool) -> tuple[pd.DataFrame, np.ndarray]:
    cols = CAT_COLS + numeric_cols(include_duration)
    if include_duration and "duration" not in df.columns:
        raise KeyError("duration stulpelio nėra, o include_duration=True")
    X = df[cols].copy()
    y = df["y_bin"].to_numpy()
    return X, y
