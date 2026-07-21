"""Walk-forward hyperparameter sweep for the xG-blend Dixon-Coles.

Two 1-D sweeps (full grid is compute we don't need): the goals/xG blend
`alpha` at default decay, then the decay rate `xi` at the winning alpha.
Every configuration is scored by the same walk-forward protocol as the
main backtest — no in-sample selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .backtest import REPORTS_DIR, _brier_logloss
from .markets import implied_probs
from .model import DEFAULT_XI
from .xgdc import XgDixonColes

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75]
XI_GRID = [0.001, 0.0019, 0.003]


def _walkforward(results: pd.DataFrame, alpha: float, xi: float,
                 seasons: int, refit_days: int) -> dict:
    end = results["Date"].max() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * seasons))
    probs, outs = [], []
    for league in data.LEAGUES:
        pool = results[results["Div"].isin(data.training_divisions(league))]
        eval_m = results[(results["Div"] == league) & (results["Date"] >= start)]
        model = XgDixonColes(xi=xi, alpha=alpha)
        t = start
        while t < end:
            t_next = t + pd.Timedelta(days=refit_days)
            window = eval_m[(eval_m["Date"] >= t) & (eval_m["Date"] < t_next)]
            if not window.empty:
                model.fit(pool, as_of=t.date())
                for _, r in window.iterrows():
                    if implied_probs(r) is None:
                        continue
                    probs.append(model.predict(r["HomeTeam"], r["AwayTeam"]))
                    outs.append(
                        0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
                    )
            t = t_next
    brier, logloss = _brier_logloss(np.array(probs), np.array(outs))
    return {
        "alpha": alpha,
        "xi": xi,
        "matches": len(outs),
        "brier": round(brier, 4),
        "logloss": round(logloss, 4),
    }


def run(seasons: int = 3, refit_days: int = 28) -> pd.DataFrame:
    results = data.load_results()
    rows = []

    for alpha in ALPHA_GRID:
        row = _walkforward(results, alpha, DEFAULT_XI, seasons, refit_days)
        rows.append(row)
        print(f"alpha={alpha} xi={DEFAULT_XI}: brier={row['brier']} logloss={row['logloss']}")
    best_alpha = min(rows, key=lambda r: r["logloss"])["alpha"]

    for xi in XI_GRID:
        if xi == DEFAULT_XI:
            continue
        row = _walkforward(results, best_alpha, xi, seasons, refit_days)
        rows.append(row)
        print(f"alpha={best_alpha} xi={xi}: brier={row['brier']} logloss={row['logloss']}")

    out = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "sweep.csv", index=False)
    print("\n" + out.to_string(index=False))
    return out
