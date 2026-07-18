"""Model registry and the weighted ensemble.

Component models implement fit(matches, as_of) -> self and
predict(home, away) -> (p_home, p_draw, p_away). The ensemble is a
log-linear pool over component probabilities whose weights are fit on
out-of-sample component predictions only — in the backtest that means
predictions from earlier walk-forward windows, live it means the
committed backtest report — so weighting never sees in-sample fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .elo import EloPoisson
from .gbm import GradientBoosted
from .model import DixonColes

ENSEMBLE_NAME = "ensemble"

# The model used for headline stats, live tables, and single-model charts.
# Chosen by backtest — see reports/backtest_summary.csv.
PRIMARY_MODEL = ENSEMBLE_NAME

KEY = ["Div", "Date", "HomeTeam", "AwayTeam"]
P_COLS = ["p_home", "p_draw", "p_away"]


def build_components() -> list:
    return [DixonColes(), EloPoisson(), GradientBoosted()]


def fit_components(components: list, matches: pd.DataFrame, as_of=None) -> None:
    for m in components:
        m.fit(matches, as_of=as_of)


def combine(probs: list[tuple[float, float, float]], weights: np.ndarray) -> tuple[float, float, float]:
    """Weighted log-linear pool of component probability triples."""
    logs = np.zeros(3)
    for w, p in zip(weights, probs):
        logs += w * np.log(np.clip(p, 1e-12, None))
    out = np.exp(logs - logs.max())
    out /= out.sum()
    return float(out[0]), float(out[1]), float(out[2])


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def fit_pool_weights(
    oos: pd.DataFrame, model_names: list[str], min_matches: int = 300
) -> np.ndarray:
    """Fit pool weights by minimizing log loss over out-of-sample component
    predictions (long format: model, KEY, p_*, FTHG, FTAG). Falls back to
    equal weights when there isn't enough history."""
    n = len(model_names)
    needed = ["model", *KEY, *P_COLS, "FTHG", "FTAG"]
    if oos.empty or any(c not in oos.columns for c in needed):
        return equal_weights(n)
    oos = oos[oos["model"].isin(model_names)].dropna(subset=P_COLS + ["FTHG", "FTAG"])
    if oos.empty:
        return equal_weights(n)

    wide = oos.pivot_table(index=KEY + ["FTHG", "FTAG"], columns="model", values=P_COLS)
    wide = wide.dropna()
    if len(wide) < min_matches:
        return equal_weights(n)

    # logp[i, k, m]: match i, outcome k, model m
    logp = np.stack(
        [
            np.log(np.clip(wide[[(c, m) for c in P_COLS]].to_numpy(dtype=float), 1e-12, None))
            for m in model_names
        ],
        axis=2,
    )
    fthg = wide.index.get_level_values("FTHG").to_numpy()
    ftag = wide.index.get_level_values("FTAG").to_numpy()
    outcome = np.select([fthg > ftag, fthg == ftag], [0, 1], 2)
    idx = np.arange(len(wide))

    def nll(z: np.ndarray) -> float:
        w = np.exp(z - z.max())
        w /= w.sum()
        pooled = (logp * w).sum(axis=2)
        pooled -= np.log(np.exp(pooled).sum(axis=1, keepdims=True))
        return -pooled[idx, outcome].mean()

    res = minimize(nll, np.zeros(n), method="Nelder-Mead")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()
