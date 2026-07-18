"""Score issued predictions against results and the closing line."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .markets import add_market_columns
from .predict import load_all_predictions

PROB_COLS = {"model": ["p_home", "p_draw", "p_away"], "market": ["mkt_home", "mkt_draw", "mkt_away"]}


def _scores(probs: np.ndarray, outcome_idx: np.ndarray) -> tuple[float, float]:
    """(brier, log_loss) for an (n, 3) probability array."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcome_idx] = 1
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    picked = np.clip(probs[np.arange(len(probs)), outcome_idx], 1e-12, None)
    logloss = float(-np.log(picked).mean())
    return brier, logloss


def run() -> pd.DataFrame:
    preds = load_all_predictions()
    if preds.empty:
        raise SystemExit("No predictions issued yet.")

    results = data.load_results()
    results = add_market_columns(results)
    results["Date"] = results["Date"].dt.date.astype(str)
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.date.astype(str)

    merged = preds.merge(
        results, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner"
    )
    if merged.empty:
        raise SystemExit("No predicted matches have results yet.")

    outcome = np.select(
        [merged["FTHG"] > merged["FTAG"], merged["FTHG"] == merged["FTAG"]], [0, 1], 2
    )

    rows = []
    for div, group_idx in [("ALL", merged.index)] + [
        (d, merged.index[merged["Div"] == d]) for d in sorted(merged["Div"].unique())
    ]:
        g = merged.loc[group_idx]
        out = outcome[merged.index.get_indexer(group_idx)]
        row = {"league": div, "matches": len(g)}
        for name, cols in PROB_COLS.items():
            sub = g.dropna(subset=cols)
            if sub.empty:
                row[f"{name}_brier"], row[f"{name}_logloss"] = None, None
                continue
            sub_out = out[g.index.get_indexer(sub.index)]
            b, ll = _scores(sub[cols].to_numpy(dtype=float), sub_out)
            row[f"{name}_brier"], row[f"{name}_logloss"] = round(b, 4), round(ll, 4)
        rows.append(row)

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    return report
