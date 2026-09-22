"""4 etapas: metrikos, grafikai, bootstrap, klaidų analizė, kaštai."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

SEED = 42
C_CALL = 1.0
V_SUCCESS = 10.0
OUTPUT_DIR = Path("outputs")


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Iš k klientų su didžiausia p̂ – kokia dalis iš tikrųjų sutiko."""
    k = int(min(k, len(y_true)))
    if k <= 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    return float(y_true[order][:k].mean())


def contact_cost(y_true: np.ndarray, y_hat: np.ndarray, c_call: float = C_CALL, v_success: float = V_SUCCESS) -> float:
    """Kaštai = c_call·(TP+FP) − v_success·TP  (mažiau = geriau; neigiama = pelnas)."""
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    return float(c_call * (tp + fp) - v_success * tp)


def best_threshold_by_cost(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Ant validation parenkame slenkstį, kuris mažina kontaktų kaštus."""
    candidates = np.unique(np.quantile(p_val, np.linspace(0.05, 0.95, 19)))
    best_t, best_c = 0.5, np.inf
    for t in candidates:
        c = contact_cost(y_val, (p_val >= t).astype(int))
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def compute_metrics(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    threshold: float,
    k: int | None = None,
    c_call: float = C_CALL,
    v_success: float = V_SUCCESS,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat, dtype=float)
    if k is None:
        k = max(1, int(0.10 * len(y_true)))
    y_bin = (p_hat >= threshold).astype(int)
    return {
        "PR-AUC": float(average_precision_score(y_true, p_hat)),
        "precision@k": precision_at_k(y_true, p_hat, k),
        "k": int(k),
        "Brier": float(brier_score_loss(y_true, p_hat)),
        "kaštai": contact_cost(y_true, y_bin, c_call, v_success),
        "slenkstis": float(threshold),
        "n": int(len(y_true)),
        "teigiamų_dalis": float(y_true.mean()),
    }


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, save_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
    plt.xlabel("Atgaminimas (Recall)")
    plt.ylabel("Tikslumas (Precision)")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=140)
    plt.show()
    print(f"Išsaugota: {path}")


def plot_calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, save_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, (y, p) in curves.items():
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        plt.plot(mean_p, frac, marker="o", label=f"{name} (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="ideali kalibracija")
    plt.xlabel("Vidutinė prognozuota p̂")
    plt.ylabel("Stebėta teigiamų dalis")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    plt.savefig(path, dpi=140)
    plt.show()
    print(f"Išsaugota: {path}")


def bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_baseline: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict:
    """
    Bootstrap testas: Δ = PR-AUC(modelis) − PR-AUC(baseline).
    p-reikšmė dvišalė H0: Δ=0.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_model = np.asarray(p_model)
    p_baseline = np.asarray(p_baseline)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = average_precision_score(y_true[idx], p_model[idx]) - average_precision_score(
            y_true[idx], p_baseline[idx]
        )
    mean_diff = float(diffs.mean())
    p_two = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_two = min(p_two, 1.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta_mean": mean_diff,
        "p_value": p_two,
        "ci95": (float(lo), float(hi)),
        "hypothesis_pass": bool(mean_diff >= 0.03 and p_two < 0.05),
    }


def save_predictions(name: str, y_true: np.ndarray, p_hat: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"preds_{name}.csv"
    pd.DataFrame({"y": y_true, "p": p_hat}).to_csv(path, index=False)
    print(f"Prognozės: {path}")
    return path


def try_load_other_predictions(exclude: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Jei Colab paleisti keli notebook'ai, čia surenkame jų preds_*.csv palyginimui."""
    found = {}
    if not OUTPUT_DIR.exists():
        return found
    for path in sorted(OUTPUT_DIR.glob("preds_*.csv")):
        key = path.stem.replace("preds_", "")
        if key == exclude:
            continue
        tab = pd.read_csv(path)
        if {"y", "p"} <= set(tab.columns):
            found[key] = (tab["y"].to_numpy(), tab["p"].to_numpy())
    return found
