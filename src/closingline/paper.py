"""Pre-registered paper trading. No real money — ever.

Each day, compare the primary model's frozen forecasts to the odds posted
in the fixtures feed. Where the model sees positive expected value beyond
a threshold, log a hypothetical quarter-Kelly bet to paper/bets.csv,
committed before kickoff like every forecast. Settlement scores each bet
against the result and the de-vigged closing line, so the repo accumulates
a tamper-evident answer to "does the model find value?" — measured in both
profit and closing-line value.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .markets import CLOSING_SOURCES, implied_probs
from .predict import load_all_predictions
from .zoo import PRIMARY_MODEL

PAPER_DIR = Path("paper")
BETS_FILE = PAPER_DIR / "bets.csv"

EV_THRESHOLD = 0.03  # only log edges > 3% — below that it's vig noise
KELLY_FRACTION = 0.25
MAX_ODDS = 8.0  # longshot region is where model tail errors concentrate

OUTCOMES = ["home", "draw", "away"]
ODDS_COLS = {"home": "B365H", "draw": "B365D", "away": "B365A"}

KEY = ["Div", "Date", "HomeTeam", "AwayTeam"]


def log_bets() -> pd.DataFrame:
    """Scan today's frozen forecasts vs current fixture odds; append new
    hypothetical bets. A fixture is only ever logged once."""
    preds = load_all_predictions()
    if preds.empty:
        return pd.DataFrame()
    preds = preds[preds["model"] == PRIMARY_MODEL].copy()
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.date.astype(str)

    fixtures = data.load_fixtures().copy()
    if fixtures.empty:
        return pd.DataFrame()
    fixtures["Date"] = fixtures["Date"].dt.date.astype(str)

    merged = preds.merge(fixtures, on=KEY, how="inner")

    existing = pd.read_csv(BETS_FILE) if BETS_FILE.exists() else pd.DataFrame()
    already = (
        set(map(tuple, existing[KEY].astype(str).to_numpy())) if not existing.empty else set()
    )

    logged_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, r in merged.iterrows():
        if tuple(str(r[k]) for k in KEY) in already:
            continue
        best = None
        for outcome in OUTCOMES:
            odds = r.get(ODDS_COLS[outcome])
            p = r[f"p_{outcome}"]
            if pd.isna(odds) or odds <= 1 or odds > MAX_ODDS:
                continue
            ev = p * odds - 1
            if ev > EV_THRESHOLD and (best is None or ev > best["ev"]):
                kelly = (p * odds - 1) / (odds - 1)
                best = {
                    "outcome": outcome,
                    "odds_taken": float(odds),
                    "p_model": float(p),
                    "ev": round(float(ev), 4),
                    "stake": round(float(KELLY_FRACTION * kelly), 4),
                }
        if best:
            rows.append({**{k: r[k] for k in KEY}, **best, "logged_at": logged_at})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    PAPER_DIR.mkdir(exist_ok=True)
    combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
    combined.to_csv(BETS_FILE, index=False)
    return out


def settle() -> pd.DataFrame | None:
    """Score settled bets: profit at quarter-Kelly stakes, plus closing-line
    value (did the close move our way?)."""
    if not BETS_FILE.exists():
        print("No bets logged yet.")
        return None
    bets = pd.read_csv(BETS_FILE)
    results = data.load_results().copy()
    results["Date"] = results["Date"].dt.date.astype(str)
    merged = bets.merge(results, on=KEY, how="inner", suffixes=("", "_r"))
    if merged.empty:
        print(f"{len(bets)} bets logged, none settled yet.")
        return None

    outcome_idx = np.select(
        [merged["FTHG"] > merged["FTAG"], merged["FTHG"] == merged["FTAG"]], [0, 1], 2
    )
    picked_idx = merged["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    won = picked_idx == outcome_idx
    merged["pnl"] = np.where(
        won, merged["stake"] * (merged["odds_taken"] - 1), -merged["stake"]
    )

    clv = []
    for _, r in merged.iterrows():
        close = implied_probs(r, CLOSING_SOURCES)
        if close is None:
            clv.append(np.nan)
            continue
        p_close = close[OUTCOMES.index(r["outcome"])]
        # Positive when the closing price is shorter than the price we took.
        clv.append(r["odds_taken"] * p_close - 1)
    merged["clv"] = clv

    summary = {
        "bets_settled": len(merged),
        "hit_rate": round(float(won.mean()), 4),
        "total_staked": round(float(merged["stake"].sum()), 4),
        "pnl_units": round(float(merged["pnl"].sum()), 4),
        "roi": round(float(merged["pnl"].sum() / merged["stake"].sum()), 4),
        "mean_clv": round(float(np.nanmean(merged["clv"])), 4),
        "positive_clv_rate": round(float((merged["clv"].dropna() > 0).mean()), 4),
    }
    PAPER_DIR.mkdir(exist_ok=True)
    merged.drop(columns=[c for c in merged.columns if c.endswith("_r")]).to_csv(
        PAPER_DIR / "settled.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(PAPER_DIR / "summary.csv", index=False)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return merged
