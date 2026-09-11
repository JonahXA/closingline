"""Pre-registered paper trading. No real money — ever.

This is a MEASUREMENT INSTRUMENT, not a strategy we expect to profit. The
backtest is unambiguous that it should not: simulated against fair closing
prices the same rule loses 10.8% per unit at a 3% edge filter, and loses
MORE as the filter tightens (-33.8% at 50%). Larger model-market
disagreement predicts larger model error, not market error. Anyone reading
paper/summary.csv should expect a negative ROI, and its value is that the
number is pre-registered and tamper-evident rather than favourable.

What it is for: closing-line value. Each day we compare the primary
model's frozen forecasts to posted odds, log the hypothetical quarter-Kelly
position the model implies, and later score it against the result AND the
closing line. If the close were to move systematically toward our
positions, that would be genuine evidence of information — the one
outcome that would reopen the edge question. The backtest says it does
not (sign agreement 47%, a coin flip), so this runs live to test that
conclusion on out-of-sample data rather than to chase profit.

The threshold is deliberately loose. It is not a tuned selection rule —
no threshold was profitable, so tightening it would only manufacture
false precision. It exists to skip the vig-noise region.
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

# Skips the vig-noise region only. NOT a tuned selection rule: backtested
# ROI is negative at every threshold and worsens as it tightens, so a
# "stricter" filter would buy false precision, not profitability.
EV_THRESHOLD = 0.03
KELLY_FRACTION = 0.25
MAX_ODDS = 8.0  # longshot region is where model tail errors concentrate

OUTCOMES = ["home", "draw", "away"]
ODDS_COLS = {"home": "B365H", "draw": "B365D", "away": "B365A"}

KEY = ["Div", "Date", "HomeTeam", "AwayTeam"]

# Pre-registered strategy set (locked 2026-09-11, BEFORE seeing results, to
# avoid multiple-comparisons p-hacking). Each is a distinct HYPOTHESIS about
# where a market inefficiency could hide, not a tuned parameter. All are
# judged on closing-line value, not ROI. A strategy is a filter: given a
# match's candidate outcomes (those clearing EV_THRESHOLD), it returns the
# outcome to back, or None to pass. `baseline` reproduces the original rule
# exactly, so its history is continuous.
def _candidates(row: pd.Series) -> list[dict]:
    """All outcomes for a match that clear the edge threshold, richest first."""
    out = []
    for outcome in OUTCOMES:
        odds = row.get(ODDS_COLS[outcome])
        p = row[f"p_{outcome}"]
        if pd.isna(odds) or odds <= 1 or odds > MAX_ODDS:
            continue
        ev = p * odds - 1
        if ev > EV_THRESHOLD:
            out.append({"outcome": outcome, "odds": float(odds), "p": float(p), "ev": float(ev)})
    return sorted(out, key=lambda c: -c["ev"])


def _pick(cands: list[dict], keep) -> dict | None:
    """First candidate (highest EV) passing the strategy's `keep` predicate."""
    for c in cands:
        if keep(c):
            return c
    return None


STRATEGIES = {
    # The original rule: back the single highest-EV outcome above 3%.
    "baseline": lambda cands: _pick(cands, lambda c: True),
    # Pickier: only very large disagreements (does selectivity help? no, per backtest).
    "selective": lambda cands: _pick(cands, lambda c: c["ev"] > 0.10),
    # Only back favorites — short-priced picks.
    "favorites": lambda cands: _pick(cands, lambda c: c["odds"] < 2.5),
    # Only back longshots — directly probes the favourite-longshot bias.
    "underdogs": lambda cands: _pick(cands, lambda c: c["odds"] > 3.5),
    # Only back draws — the outcome models handle worst.
    "draws": lambda cands: _pick(cands, lambda c: c["outcome"] == "draw"),
}


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
    # Bets logged before the multi-strategy split carry no `strategy` column;
    # they are the baseline history and are tagged as such.
    if not existing.empty and "strategy" not in existing.columns:
        existing["strategy"] = "baseline"
    already = set()
    if not existing.empty:
        already = set(
            map(tuple, existing[["strategy", *KEY]].astype(str).to_numpy())
        )

    logged_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, r in merged.iterrows():
        cands = _candidates(r)
        if not cands:
            continue
        for name, strat in STRATEGIES.items():
            if (name, *(str(r[k]) for k in KEY)) in already:
                continue
            pick = strat(cands)
            if pick is None:
                continue
            kelly = (pick["p"] * pick["odds"] - 1) / (pick["odds"] - 1)
            rows.append(
                {
                    "strategy": name,
                    **{k: r[k] for k in KEY},
                    "outcome": pick["outcome"],
                    "odds_taken": pick["odds"],
                    "p_model": round(pick["p"], 4),
                    "ev": round(pick["ev"], 4),
                    "stake": round(float(KELLY_FRACTION * kelly), 4),
                    "logged_at": logged_at,
                }
            )

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

    if "strategy" not in merged.columns:
        merged["strategy"] = "baseline"

    outcome_idx = np.select(
        [merged["FTHG"] > merged["FTAG"], merged["FTHG"] == merged["FTAG"]], [0, 1], 2
    )
    picked_idx = merged["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    merged["won"] = picked_idx == outcome_idx
    merged["pnl"] = np.where(
        merged["won"], merged["stake"] * (merged["odds_taken"] - 1), -merged["stake"]
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

    def _summarize(g: pd.DataFrame) -> dict:
        return {
            "bets_settled": len(g),
            "hit_rate": round(float(g["won"].mean()), 4),
            "total_staked": round(float(g["stake"].sum()), 4),
            "pnl_units": round(float(g["pnl"].sum()), 4),
            "roi": round(float(g["pnl"].sum() / g["stake"].sum()), 4) if g["stake"].sum() else 0.0,
            "mean_clv": round(float(np.nanmean(g["clv"])), 4) if g["clv"].notna().any() else None,
            "positive_clv_rate": round(float((g["clv"].dropna() > 0).mean()), 4)
            if g["clv"].notna().any() else None,
        }

    PAPER_DIR.mkdir(exist_ok=True)
    merged.drop(columns=[c for c in merged.columns if c.endswith("_r")]).to_csv(
        PAPER_DIR / "settled.csv", index=False
    )

    # Per-strategy board (pre-registered comparison, judged on CLV).
    board = pd.DataFrame(
        [{"strategy": s, **_summarize(g)} for s, g in merged.groupby("strategy")]
    ).sort_values("strategy")
    board.to_csv(PAPER_DIR / "strategies.csv", index=False)

    # summary.csv stays the baseline strategy for dashboard/history continuity.
    base = merged[merged["strategy"] == "baseline"]
    summary = _summarize(base) if not base.empty else _summarize(merged)
    pd.DataFrame([summary]).to_csv(PAPER_DIR / "summary.csv", index=False)

    print(board.to_string(index=False))
    return merged
