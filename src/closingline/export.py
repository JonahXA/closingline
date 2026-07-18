"""Export a single JSON data file for the public dashboard."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .backtest import REPORTS_DIR, summarize
from .markets import add_market_columns
from .predict import load_all_predictions
from .zoo import PRIMARY_MODEL

DEST = Path("dashboard/public/data.json")

N_BINS = 10


def _outcome(df: pd.DataFrame) -> np.ndarray:
    return np.select([df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], [0, 1], 2)


def _calibration(bt: pd.DataFrame) -> list[dict]:
    """Pooled reliability curve: every outcome probability vs its indicator."""
    outcome = _outcome(bt)
    probs = bt[["p_home", "p_draw", "p_away"]].to_numpy(float).ravel()
    hits = np.zeros((len(bt), 3))
    hits[np.arange(len(bt)), outcome] = 1
    hits = hits.ravel()

    bins = np.clip((probs * N_BINS).astype(int), 0, N_BINS - 1)
    out = []
    for b in range(N_BINS):
        mask = bins == b
        if mask.sum() < 10:
            continue
        out.append(
            {
                "bin_mid": round((b + 0.5) / N_BINS, 2),
                "predicted": round(float(probs[mask].mean()), 4),
                "observed": round(float(hits[mask].mean()), 4),
                "n": int(mask.sum()),
            }
        )
    return out


def _monthly(bt: pd.DataFrame) -> list[dict]:
    bt = bt.dropna(subset=["mkt_home"]).copy()
    bt["month"] = pd.to_datetime(bt["Date"]).dt.to_period("M").astype(str)
    outcome = _outcome(bt)
    onehot = np.zeros((len(bt), 3))
    onehot[np.arange(len(bt)), outcome] = 1
    for name, cols in [("model", ["p_home", "p_draw", "p_away"]),
                       ("market", ["mkt_home", "mkt_draw", "mkt_away"])]:
        bt[f"{name}_sq"] = ((bt[cols].to_numpy(float) - onehot) ** 2).sum(axis=1)
    g = bt.groupby("month").agg(
        model_brier=("model_sq", "mean"), market_brier=("market_sq", "mean"), n=("Div", "size")
    )
    return [
        {"month": m, "model_brier": round(r.model_brier, 4),
         "market_brier": round(r.market_brier, 4), "n": int(r.n)}
        for m, r in g.iterrows()
    ]


def run() -> None:
    payload: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "leagues": data.LEAGUES,
        "backtest": None,
        "live": {"upcoming": [], "scored": []},
    }

    bt_path = REPORTS_DIR / "backtest.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        if "model" not in bt.columns:
            bt = bt.assign(model=PRIMARY_MODEL)
        summary = summarize(bt)
        primary = bt[bt["model"] == PRIMARY_MODEL]
        payload["backtest"] = {
            "primary_model": PRIMARY_MODEL,
            # Per-league rows for the primary model drive the league charts;
            # the ALL rows across models drive the model-zoo comparison.
            "summary": summary[summary["model"] == PRIMARY_MODEL].to_dict(orient="records"),
            "models": summary[summary["league"] == "ALL"].to_dict(orient="records"),
            "calibration": _calibration(primary),
            "monthly": _monthly(primary),
            "start": bt["Date"].min(),
            "end": bt["Date"].max(),
        }

    preds = load_all_predictions()
    if not preds.empty:
        preds = preds[preds["model"] == PRIMARY_MODEL]
    if not preds.empty:
        preds["Date"] = pd.to_datetime(preds["Date"]).dt.date.astype(str)
        results = add_market_columns(data.load_results())
        results["Date"] = results["Date"].dt.date.astype(str)
        merged = preds.merge(
            results[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                     "mkt_home", "mkt_draw", "mkt_away"]],
            on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left",
        )
        played = merged["FTHG"].notna()
        payload["live"]["upcoming"] = json.loads(
            merged[~played].drop(columns=["FTHG", "FTAG"]).to_json(orient="records")
        )
        scored = merged[played].copy()
        payload["live"]["scored"] = json.loads(scored.to_json(orient="records"))
        if len(scored) >= 20:
            scored_bt = scored.rename(columns={})
            payload["live"]["summary"] = summarize(scored_bt).to_dict(orient="records")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {DEST} ({DEST.stat().st_size / 1024:.0f} KB)")
