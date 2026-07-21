"""Significance tests on per-match score differentials.

Reporting "model Brier 0.5854 vs market 0.5734" invites the obvious
question — is that gap distinguishable from noise? This answers it two
ways, both operating on the paired, per-match score differences d_i =
score_model_i - score_other_i:

  * Paired bootstrap: resample matches with replacement, giving a
    distribution-free CI on the mean differential and a two-sided p-value.
  * Diebold-Mariano: the standard test for comparing forecast accuracy,
    with a Newey-West (HAC) variance so same-round correlation between
    match scores doesn't understate the standard error.

A positive mean differential means the first forecaster scored higher
(worse, for Brier/log loss) than the second. We test model-vs-market and
the xG-blend-vs-goals improvement — the two claims the project leans on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import REPORTS_DIR
from .zoo import PRIMARY_MODEL

P_COLS = ["p_home", "p_draw", "p_away"]
M_COLS = ["mkt_home", "mkt_draw", "mkt_away"]
RNG = np.random.default_rng(20260721)
N_BOOT = 10000


def _outcome(df: pd.DataFrame) -> np.ndarray:
    return np.select([df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], [0, 1], 2)


def _brier_per_match(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcome] = 1
    return ((probs - onehot) ** 2).sum(axis=1)


def _logloss_per_match(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    picked = np.clip(probs[np.arange(len(probs)), outcome], 1e-12, None)
    return -np.log(picked)


def _diebold_mariano(d: np.ndarray) -> tuple[float, float]:
    """DM statistic and two-sided p-value for mean(d) = 0, with a
    Newey-West HAC variance (lag = n^{1/3})."""
    n = len(d)
    dbar = d.mean()
    demeaned = d - dbar
    lag = max(1, int(round(n ** (1 / 3))))
    gamma0 = (demeaned @ demeaned) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)  # Bartlett kernel
        cov = (demeaned[k:] @ demeaned[:-k]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    dm = dbar / se
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def _bootstrap(d: np.ndarray) -> tuple[float, float, float]:
    """(ci_low, ci_high, two-sided p) for mean(d) via paired bootstrap."""
    n = len(d)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided bootstrap p: how often the resampled mean crosses zero.
    frac = (means <= 0).mean() if d.mean() > 0 else (means >= 0).mean()
    p = min(1.0, 2 * frac)
    return float(lo), float(hi), float(p)


def _compare(bt: pd.DataFrame, a_cols, b_cols, a_model, b_model, metric: str) -> dict:
    """Per-match differential (a - b) for one metric, on matches both
    forecasters cover, then bootstrap + DM."""
    fn = _brier_per_match if metric == "brier" else _logloss_per_match
    if a_model == b_model:  # model vs market: same rows, different columns
        g = bt[bt["model"] == a_model].dropna(subset=b_cols).reset_index(drop=True)
        out = _outcome(g)
        sa = fn(g[a_cols].to_numpy(float), out)
        sb = fn(g[b_cols].to_numpy(float), out)
    else:  # model vs model: align on the shared match key
        key = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        ga = bt[bt["model"] == a_model][key + P_COLS]
        gb = bt[bt["model"] == b_model][key + P_COLS]
        g = ga.merge(gb, on=key, suffixes=("_a", "_b"))
        out = _outcome(g)
        sa = fn(g[[f"{c}_a" for c in P_COLS]].to_numpy(float), out)
        sb = fn(g[[f"{c}_b" for c in P_COLS]].to_numpy(float), out)
    d = sa - sb
    lo, hi, p_boot = _bootstrap(d)
    dm, p_dm = _diebold_mariano(d)
    label = f"{a_model} vs closing line" if a_model == b_model else f"{a_model} vs {b_model}"
    return {
        "comparison": label,
        "metric": metric,
        "matches": len(d),
        "mean_diff": round(float(d.mean()), 5),
        "ci_low": round(lo, 5),
        "ci_high": round(hi, 5),
        "p_bootstrap": round(p_boot, 4),
        "dm_stat": round(dm, 3),
        "p_dm": round(p_dm, 4),
    }


def run() -> pd.DataFrame:
    bt = pd.read_csv(REPORTS_DIR / "backtest.csv").dropna(subset=M_COLS)
    rows = []
    for metric in ("brier", "logloss"):
        # Does the market genuinely beat our primary model?
        rows.append(_compare(bt, P_COLS, M_COLS, PRIMARY_MODEL, PRIMARY_MODEL, metric))
        # Is the xG-blend a real improvement over goals-only Dixon-Coles?
        if {"xg-dixon-coles", "dixon-coles"}.issubset(bt["model"].unique()):
            rows.append(
                _compare(bt, P_COLS, P_COLS, "xg-dixon-coles", "dixon-coles", metric)
            )
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "significance.csv", index=False)
    print(out.to_string(index=False))
    return out
