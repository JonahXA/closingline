"""Closing-line-value study.

Three questions, all answered from the committed walk-forward backtest and
the opening/closing odds already in the data:

1. How much does the market learn between opening and close?
   (Brier of the de-vigged opening line vs the closing line.)
2. Does our model beat the *opening* line, even though it loses to the close?
3. Does the model predict line movement — i.e., when the model disagrees
   with the opening line, does the close move toward the model?

This is market-efficiency research; no wagering is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .backtest import REPORTS_DIR, _brier_logloss
from .markets import CLOSING_SOURCES, OPENING_SOURCES, implied_probs
from .zoo import PRIMARY_MODEL

P_COLS = ["p_home", "p_draw", "p_away"]


def run() -> dict:
    bt = pd.read_csv(REPORTS_DIR / "backtest.csv")
    bt = bt[bt["model"] == PRIMARY_MODEL]

    results = data.load_results()
    results = results.assign(Date=results["Date"].dt.date.astype(str))
    merged = bt.merge(
        results, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner", suffixes=("", "_r")
    )

    opens, closes = [], []
    for _, r in merged.iterrows():
        opens.append(implied_probs(r, OPENING_SOURCES))
        closes.append(implied_probs(r, CLOSING_SOURCES))
    ok = [i for i, (o, c) in enumerate(zip(opens, closes)) if o and c]
    merged = merged.iloc[ok].reset_index(drop=True)
    p_open = np.array([opens[i][:3] for i in ok])
    p_close = np.array([closes[i][:3] for i in ok])
    p_model = merged[P_COLS].to_numpy(dtype=float)
    outcome = np.select(
        [merged["FTHG"] > merged["FTAG"], merged["FTHG"] == merged["FTAG"]], [0, 1], 2
    )

    scores = {
        name: _brier_logloss(p, outcome)
        for name, p in [("model", p_model), ("opening", p_open), ("closing", p_close)]
    }

    # Line-movement prediction: when the model disagrees with the opening
    # line, does the close move the same way? Pooled over all outcomes.
    model_dev = (p_model - p_open).ravel()
    close_move = (p_close - p_open).ravel()
    moved = np.abs(close_move) > 1e-9
    corr = float(np.corrcoef(model_dev[moved], close_move[moved])[0, 1])
    sign_agree = float((np.sign(model_dev[moved]) == np.sign(close_move[moved])).mean())

    out = {
        "matches": int(len(merged)),
        **{f"{k}_brier": round(v[0], 4) for k, v in scores.items()},
        **{f"{k}_logloss": round(v[1], 4) for k, v in scores.items()},
        "movement_corr": round(corr, 4),
        "movement_sign_agreement": round(sign_agree, 4),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([out]).to_csv(REPORTS_DIR / "clv.csv", index=False)
    for k, v in out.items():
        print(f"{k}: {v}")
    return out
