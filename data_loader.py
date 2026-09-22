"""1 etapas: bank-full.csv nuskaitymas ir chronologinės tvarkos patikra."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SEED = 42
DATA_PATH = "bank-full.csv"

MONTH_ORDER = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]


def load_bank(path: str = DATA_PATH) -> pd.DataFrame:
    """Nuskaito UCI Bank Marketing failą. Stulpeliai atskirti kabliataškiu."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Nerastas {path}. Colab: Files → Upload → bank-full.csv "
            "(tas pats aplankas kaip notebook)."
        )
    df = pd.read_csv(csv_path, sep=";")
    expected_cols = {
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "previous", "poutcome", "y",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Trūksta stulpelių: {missing}")
    if len(df) != 45211:
        print(f"Dėmesio: tikėtasi 45211 eilučių, gauta {len(df)}.")
    return df


def month_change_indices(df: pd.DataFrame) -> list[int]:
    """Eilučių indeksai, kur keičiasi month (mėnesio virsmo taškai)."""
    months = df["month"].to_numpy()
    changes = [0]
    for i in range(1, len(months)):
        if months[i] != months[i - 1]:
            changes.append(i)
    changes.append(len(df))
    return changes


def check_chronological_order(df: pd.DataFrame) -> None:
    """Patikrina, kad eilučių tvarka atitinka kampanijų sezoniškumą (month virsmai)."""
    changes = month_change_indices(df)
    n_runs = len(changes) - 1
    print(f"Mėnesio-virsmo taškų (periodų) skaičius: {n_runs}")
    print("Pirmi 5 periodai:")
    for i, start in enumerate(changes[:-1][:5]):
        end = changes[i + 1]
        print(f"  {start:5d}–{end:5d}  month={df.iloc[start]['month']}  n={end - start}")
    if n_runs < 20:
        print("Įspėjimas: mažiau periodų nei dokumente (~30).")
