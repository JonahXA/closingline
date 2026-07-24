"""Walk-forward hyperparameter sweeps.

Coordinate-descent style 1-D sweeps (a full grid is compute we don't
need): vary one parameter at a time, keep the winner, move on. Every
configuration is scored by the same walk-forward protocol as the main
backtest — refit on a rolling window, predict only forward — so nothing
here is selected in-sample.

`sweep` tunes the xG-blend Dixon-Coles (blend ratio, then decay rate).
`sweep --model gbm` and `--model elo` tune the two components whose
hyperparameters were set by intuition on day one and never validated.
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


def _walkforward_model(results: pd.DataFrame, factory, seasons: int,
                       refit_days: int) -> tuple[float, float, int]:
    """Score a model built by `factory()` under the standard walk-forward
    protocol. Returns (brier, logloss, n_matches)."""
    end = results["Date"].max() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * seasons))
    probs, outs = [], []
    for league in data.LEAGUES:
        pool = results[results["Div"].isin(data.training_divisions(league))]
        eval_m = results[(results["Div"] == league) & (results["Date"] >= start)]
        model = factory()
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
    return brier, logloss, len(outs)


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


def _coordinate_descent(results, factory, base: dict, grids: dict[str, list],
                        seasons: int, refit_days: int, label: str) -> pd.DataFrame:
    """Vary one parameter at a time, keeping the best value found so far.
    The baseline config is scored first so any 'improvement' is measured
    against the values currently in production."""
    rows = []
    b, ll, n = _walkforward_model(results, lambda: factory(**base), seasons, refit_days)
    rows.append({**base, "brier": round(b, 4), "logloss": round(ll, 4), "matches": n,
                 "note": "baseline (current production values)"})
    print(f"baseline {base}: brier={b:.4f} logloss={ll:.4f}")
    best = dict(base)
    best_ll = ll

    for param, grid in grids.items():
        for value in grid:
            if value == best[param]:
                continue
            cand = {**best, param: value}
            b, ll, n = _walkforward_model(
                results, lambda c=cand: factory(**c), seasons, refit_days
            )
            rows.append({**cand, "brier": round(b, 4), "logloss": round(ll, 4),
                         "matches": n, "note": f"vary {param}"})
            print(f"  {param}={value}: brier={b:.4f} logloss={ll:.4f}")
            if ll < best_ll:
                best_ll, best = ll, cand
        print(f"-> best after {param}: {best[param]} (logloss {best_ll:.4f})")

    out = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / f"sweep_{label}.csv", index=False)
    print("\n" + out.to_string(index=False))
    return out


def run_gbm(seasons: int = 3, refit_days: int = 28) -> pd.DataFrame:
    """Tune the gradient-booster's tree/regularization settings."""
    from .gbm import GBM_PARAMS, GradientBoosted

    results = data.load_results()
    base = {k: GBM_PARAMS[k] for k in
            ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
    grids = {
        "learning_rate": [0.03, 0.06, 0.10],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [20, 40, 80],
        "l2_regularization": [0.1, 1.0, 5.0],
    }
    return _coordinate_descent(
        results, GradientBoosted, base, grids, seasons, refit_days, "gbm"
    )


def run_shrinkage(seasons: int = 3, refit_days: int = 28) -> pd.DataFrame:
    """Tune empirical-Bayes shrinkage on the dominant xG-Dixon-Coles.

    Targets the documented season-opening weakness: with few matches, MLE
    team ratings are noisy, and pulling them toward the league mean should
    help most exactly when the model is weakest.
    """
    from .xgdc import XGDC_ALPHA, XGDC_XI

    results = data.load_results()
    base = {"xi": XGDC_XI, "alpha": XGDC_ALPHA, "shrinkage": 0.0}
    grids = {"shrinkage": [0.0, 0.5, 2.0, 5.0, 15.0]}
    return _coordinate_descent(
        results, XgDixonColes, base, grids, seasons, refit_days, "shrinkage"
    )


def run_elo(seasons: int = 3, refit_days: int = 28) -> pd.DataFrame:
    """Tune Elo's update rate and home-field constant."""
    from .elo import EloPoisson

    results = data.load_results()
    base = {"k": 20.0, "hfa": 60.0}
    grids = {"k": [10.0, 20.0, 30.0, 45.0], "hfa": [30.0, 60.0, 90.0]}
    return _coordinate_descent(
        results, EloPoisson, base, grids, seasons, refit_days, "elo"
    )


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
