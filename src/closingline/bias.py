"""Scan the backtest for pockets where the market is soft.

Slices the walk-forward backtest by favorite strength, month, and league,
comparing model vs market Brier in each bucket — the systematic version of
"is there anywhere we're already competitive?". Also simulates the paper
strategy against historical *closing* odds (the hardest possible prices)
across EV thresholds, so the expected ROI of the live paper tracker is
known before it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import REPORTS_DIR, _brier_logloss
from .zoo import PRIMARY_MODEL

P_COLS = ["p_home", "p_draw", "p_away"]
M_COLS = ["mkt_home", "mkt_draw", "mkt_away"]


def _load() -> tuple[pd.DataFrame, np.ndarray]:
    bt = pd.read_csv(REPORTS_DIR / "backtest.csv")
    bt = bt[(bt["model"] == PRIMARY_MODEL)].dropna(subset=M_COLS).reset_index(drop=True)
    outcome = np.select([bt["FTHG"] > bt["FTAG"], bt["FTHG"] == bt["FTAG"]], [0, 1], 2)
    return bt, outcome


def _bucket_rows(bt: pd.DataFrame, outcome: np.ndarray, label_col: pd.Series, dim: str) -> list[dict]:
    rows = []
    for label in sorted(label_col.unique()):
        mask = (label_col == label).to_numpy()
        if mask.sum() < 100:
            continue
        g, out = bt[mask], outcome[mask]
        mb, _ = _brier_logloss(g[P_COLS].to_numpy(float), out)
        kb, _ = _brier_logloss(g[M_COLS].to_numpy(float), out)
        rows.append(
            {
                "dimension": dim,
                "bucket": str(label),
                "matches": int(mask.sum()),
                "model_brier": round(mb, 4),
                "market_brier": round(kb, 4),
                "gap": round(mb - kb, 4),
            }
        )
    return rows


def run() -> pd.DataFrame:
    bt, outcome = _load()

    fav = bt[M_COLS].max(axis=1)
    fav_bucket = pd.cut(
        fav, [0, 0.45, 0.55, 0.65, 0.75, 1.0],
        labels=["<45% fav", "45-55%", "55-65%", "65-75%", ">75% fav"],
    )
    month = pd.to_datetime(bt["Date"]).dt.month.map(
        lambda m: "Aug-Sep" if m in (8, 9) else ("Oct-Dec" if m in (10, 11, 12) else "Jan-May")
    )

    rows = (
        _bucket_rows(bt, outcome, fav_bucket.astype(str), "favorite_strength")
        + _bucket_rows(bt, outcome, month, "period")
        + _bucket_rows(bt, outcome, bt["Div"], "league")
    )
    scan = pd.DataFrame(rows).sort_values(["dimension", "gap"])

    # Hypothetical value betting vs de-vigged closing prices — the purest
    # information test (no vig, sharpest prices; live prices would be softer).
    close_odds = 1.0 / bt[M_COLS].to_numpy(float)
    p_model = bt[P_COLS].to_numpy(float)
    sims = []
    for threshold in (0.02, 0.05, 0.10):
        ev = p_model * close_odds - 1
        pick = ev.argmax(axis=1)
        picked_ev = ev[np.arange(len(bt)), pick]
        take = picked_ev > threshold
        if take.sum() < 30:
            continue
        odds_taken = close_odds[np.arange(len(bt)), pick][take]
        won = (pick == outcome)[take]
        pnl = np.where(won, odds_taken - 1, -1.0)
        sims.append(
            {
                "edge_threshold": f">{threshold:.0%}",
                "bets": int(take.sum()),
                "hit_rate": round(float(won.mean()), 4),
                "avg_odds": round(float(np.mean(odds_taken)), 4),
                "roi_per_unit": round(float(pnl.mean()), 4),
            }
        )
    ev_sim = pd.DataFrame(sims)

    REPORTS_DIR.mkdir(exist_ok=True)
    scan.to_csv(REPORTS_DIR / "bias_scan.csv", index=False)
    ev_sim.to_csv(REPORTS_DIR / "ev_sim.csv", index=False)
    print(scan.to_string(index=False))
    print("\nEV simulation vs de-vigged closing prices:")
    print(ev_sim.to_string(index=False))
    return scan
