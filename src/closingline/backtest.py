"""Walk-forward historical backtest across the model zoo.

Steps through past seasons refitting every `refit_days`, predicting each
top-flight match using only data available before the refit date — the
same information regime as the live pipeline, so backtest numbers are an
honest preview of live performance. Output is long format: one row per
match per model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .markets import implied_probs
from .zoo import build_models, fit_all

REPORTS_DIR = Path("reports")


def run(seasons: int = 3, refit_days: int = 28) -> pd.DataFrame:
    results = data.load_results()
    end = results["Date"].max() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(365.25 * seasons))

    rows = []
    for league in data.LEAGUES:
        pool = results[results["Div"].isin(data.training_divisions(league))]
        eval_m = results[(results["Div"] == league) & (results["Date"] >= start)]
        models = build_models()
        t = start
        while t < end:
            t_next = t + pd.Timedelta(days=refit_days)
            window = eval_m[(eval_m["Date"] >= t) & (eval_m["Date"] < t_next)]
            if not window.empty:
                fit_all(models, pool, as_of=t.date())
                for _, r in window.iterrows():
                    mkt = implied_probs(r)
                    for model in models:
                        p_home, p_draw, p_away = model.predict(r["HomeTeam"], r["AwayTeam"])
                        rows.append(
                            {
                                "model": model.name,
                                "Div": league,
                                "Date": r["Date"].date().isoformat(),
                                "HomeTeam": r["HomeTeam"],
                                "AwayTeam": r["AwayTeam"],
                                "FTHG": r["FTHG"],
                                "FTAG": r["FTAG"],
                                "p_home": round(p_home, 4),
                                "p_draw": round(p_draw, 4),
                                "p_away": round(p_away, 4),
                                "mkt_home": round(mkt[0], 4) if mkt else None,
                                "mkt_draw": round(mkt[1], 4) if mkt else None,
                                "mkt_away": round(mkt[2], 4) if mkt else None,
                                "odds_source": mkt[3] if mkt else None,
                                "as_of": t.date().isoformat(),
                            }
                        )
            t = t_next
        print(f"{league}: done")

    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "backtest.csv", index=False)
    summary = summarize(out)
    summary.to_csv(REPORTS_DIR / "backtest_summary.csv", index=False)
    print(summary.to_string(index=False))
    return out


def _brier_logloss(probs: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcome] = 1
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    picked = np.clip(probs[np.arange(len(probs)), outcome], 1e-12, None)
    return brier, float(-np.log(picked).mean())


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Per (model, league) metrics; league 'ALL' pools every league."""
    if "model" not in bt.columns:  # backtests written before the model zoo
        bt = bt.assign(model="dixon-coles")
    bt = bt.dropna(subset=["mkt_home"]).reset_index(drop=True)
    rows = []
    for model in bt["model"].unique():
        g_m = bt[bt["model"] == model]
        outcome_m = np.select(
            [g_m["FTHG"] > g_m["FTAG"], g_m["FTHG"] == g_m["FTAG"]], [0, 1], 2
        )
        for label in ["ALL"] + sorted(g_m["Div"].unique()):
            mask = np.ones(len(g_m), dtype=bool) if label == "ALL" else (g_m["Div"] == label).to_numpy()
            g, out = g_m[mask], outcome_m[mask]
            mb, mll = _brier_logloss(g[["p_home", "p_draw", "p_away"]].to_numpy(float), out)
            kb, kll = _brier_logloss(g[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(float), out)
            rows.append(
                {
                    "model": model,
                    "league": label,
                    "matches": len(g),
                    "model_brier": round(mb, 4),
                    "market_brier": round(kb, 4),
                    "model_logloss": round(mll, 4),
                    "market_logloss": round(kll, 4),
                }
            )
    return pd.DataFrame(rows)
