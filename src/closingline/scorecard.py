"""Live-season scorecard: how the pre-registered forecasts are doing.

This is the project's payoff. The backtest showed a +2.09% Brier gap to the
closing line; the live question is whether that holds out-of-sample on
forecasts that were frozen in git *before* kickoff. Only forecasts issued
by the live pipeline (predictions/) are scored — never the backtest — so
the scorecard is the honest, unfakeable track record.

It is designed to be run whenever, including when almost nothing has
resolved yet: it reports how many live forecasts exist, how many have
results, and — once there is enough resolved history — the live Brier/log
loss for each model against the de-vigged closing line, with a
significance test on the ensemble-vs-market gap. Output is written to
reports/scorecard.csv (committed) so it accumulates over the season and
feeds the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .evaluate import _scores
from .markets import add_market_columns
from .predict import load_all_predictions
from .zoo import PRIMARY_MODEL

REPORTS_DIR = Path("reports")
PROB_COLS = ["p_home", "p_draw", "p_away"]
MKT_COLS = ["mkt_home", "mkt_draw", "mkt_away"]

# Below this many resolved matches, per-model Brier is too noisy to report
# as a headline — we still show counts and a provisional figure, clearly
# flagged, but withhold significance claims.
MIN_FOR_SIGNIFICANCE = 50


def _resolved() -> pd.DataFrame:
    """Live pre-registered forecasts joined to results + de-vigged market
    prices, one row per (match, model). Empty frame if nothing has resolved."""
    preds = load_all_predictions()
    if preds.empty:
        return preds
    preds["Date"] = pd.to_datetime(preds["Date"]).dt.date.astype(str)

    results = add_market_columns(data.load_results())
    results["Date"] = results["Date"].dt.date.astype(str)
    keep = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *MKT_COLS]
    return preds.merge(results[keep], on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner")


def run() -> dict:
    preds = load_all_predictions()
    n_forecasts = 0 if preds.empty else preds.drop_duplicates(
        ["Div", "Date", "HomeTeam", "AwayTeam"]
    ).shape[0]

    merged = _resolved()
    status = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "live_matches_forecast": int(n_forecasts),
        "live_matches_resolved": 0,
        "primary_model": PRIMARY_MODEL,
        "enough_data": False,
    }

    if merged.empty:
        print(
            f"{n_forecasts} live forecasts issued; none resolved yet.\n"
            "The scorecard fills in as matches are played."
        )
        _write(status, pd.DataFrame())
        return status

    resolved_matches = merged.drop_duplicates(["Div", "Date", "HomeTeam", "AwayTeam"]).shape[0]
    status["live_matches_resolved"] = int(resolved_matches)
    status["enough_data"] = resolved_matches >= MIN_FOR_SIGNIFICANCE

    rows = []
    for model in sorted(merged["model"].unique()):
        gm = merged[merged["model"] == model].dropna(subset=MKT_COLS).reset_index(drop=True)
        if gm.empty:
            continue
        outcome = np.select([gm["FTHG"] > gm["FTAG"], gm["FTHG"] == gm["FTAG"]], [0, 1], 2)
        mb, mll = _scores(gm[PROB_COLS].to_numpy(float), outcome)
        kb, kll = _scores(gm[MKT_COLS].to_numpy(float), outcome)
        rows.append(
            {
                "model": model,
                "matches": len(gm),
                "model_brier": round(mb, 4),
                "market_brier": round(kb, 4),
                "gap": round(mb - kb, 4),
                "model_logloss": round(mll, 4),
                "market_logloss": round(kll, 4),
            }
        )
    board = pd.DataFrame(rows)

    # Significance on the primary model vs the market, once there's enough.
    sig = _significance(merged) if status["enough_data"] else None
    if sig:
        status.update(sig)

    _write(status, board)
    _print(status, board, sig)
    return status


def _significance(merged: pd.DataFrame) -> dict | None:
    """Paired bootstrap + DM on primary-model vs market per-match Brier,
    reusing the same machinery as the backtest significance test."""
    from .significance import _bootstrap, _diebold_mariano

    g = merged[merged["model"] == PRIMARY_MODEL].dropna(subset=MKT_COLS).reset_index(drop=True)
    if g.empty:
        return None
    outcome = np.select([g["FTHG"] > g["FTAG"], g["FTHG"] == g["FTAG"]], [0, 1], 2)
    onehot = np.zeros((len(g), 3))
    onehot[np.arange(len(g)), outcome] = 1
    d = (
        ((g[PROB_COLS].to_numpy(float) - onehot) ** 2).sum(1)
        - ((g[MKT_COLS].to_numpy(float) - onehot) ** 2).sum(1)
    )
    lo, hi, p_boot = _bootstrap(d)
    dm, p_dm = _diebold_mariano(d)
    return {
        "sig_mean_gap": round(float(d.mean()), 5),
        "sig_ci_low": round(lo, 5),
        "sig_ci_high": round(hi, 5),
        "sig_p_bootstrap": round(p_boot, 4),
        "sig_p_dm": round(p_dm, 4),
    }


def _write(status: dict, board: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    board.to_csv(REPORTS_DIR / "scorecard.csv", index=False)
    pd.DataFrame([status]).to_csv(REPORTS_DIR / "scorecard_status.csv", index=False)


def _print(status: dict, board: pd.DataFrame, sig: dict | None) -> None:
    print(
        f"Live scorecard — {status['live_matches_resolved']} of "
        f"{status['live_matches_forecast']} pre-registered forecasts resolved"
    )
    if not board.empty:
        print(board.to_string(index=False))
    if sig:
        verdict = "market wins" if sig["sig_ci_low"] > 0 else (
            "model wins" if sig["sig_ci_high"] < 0 else "not distinguishable"
        )
        print(
            f"\nprimary ({status['primary_model']}) vs market: mean gap "
            f"{sig['sig_mean_gap']:+.4f}, 95% CI [{sig['sig_ci_low']:+.4f}, "
            f"{sig['sig_ci_high']:+.4f}], p_boot={sig['sig_p_bootstrap']} — {verdict}"
        )
    elif not board.empty:
        print(
            f"\n(only {status['live_matches_resolved']} matches resolved; "
            f"significance withheld until {MIN_FOR_SIGNIFICANCE}+ — figures are provisional)"
        )
