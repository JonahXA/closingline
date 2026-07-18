"""Generate and freeze forecasts for upcoming fixtures.

Predictions are pre-registered: once a fixture has an issued forecast in
predictions/, it is never re-predicted or overwritten. Files are named by
issuance date so the git history timestamps every forecast.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import data
from .zoo import (
    ENSEMBLE_NAME,
    build_components,
    combine,
    equal_weights,
    fit_components,
    fit_pool_weights,
)

PREDICTIONS_DIR = Path("predictions")

KEY = ["Div", "Date", "HomeTeam", "AwayTeam"]


def load_all_predictions() -> pd.DataFrame:
    files = sorted(PREDICTIONS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    out = pd.concat([pd.read_csv(f, parse_dates=["Date"]) for f in files], ignore_index=True)
    return out


def run(horizon_days: int = 7) -> pd.DataFrame:
    """Predict all not-yet-predicted fixtures within the horizon."""
    results = data.load_results()
    fixtures = data.load_fixtures()

    today = pd.Timestamp(dt.date.today())
    fixtures = fixtures[
        (fixtures["Date"] >= today)
        & (fixtures["Date"] <= today + pd.Timedelta(days=horizon_days))
    ]

    existing = load_all_predictions()
    if not existing.empty:
        issued = set(map(tuple, existing[KEY].astype(str).to_numpy()))
        mask = ~fixtures[KEY].astype(str).apply(tuple, axis=1).isin(issued)
        fixtures = fixtures[mask]

    if fixtures.empty:
        return pd.DataFrame()

    rows = []
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    # Live ensemble weights come from the committed backtest report — all
    # out-of-sample component predictions, nothing fit today.
    bt_path = Path("reports/backtest.csv")
    oos = pd.read_csv(bt_path) if bt_path.exists() else pd.DataFrame()

    for div in fixtures["Div"].unique():
        components = build_components()
        names = [c.name for c in components]
        fit_components(components, results[results["Div"].isin(data.training_divisions(div))])
        weights = fit_pool_weights(oos, names) if not oos.empty else equal_weights(len(names))
        for _, fx in fixtures[fixtures["Div"] == div].iterrows():
            probs = []
            for model in components:
                p = model.predict(fx["HomeTeam"], fx["AwayTeam"])
                probs.append(p)
            for model_name, p in list(zip(names, probs)) + [
                (ENSEMBLE_NAME, combine(probs, weights))
            ]:
                rows.append(
                    {
                        "Div": div,
                        "Date": fx["Date"].date().isoformat(),
                        "HomeTeam": fx["HomeTeam"],
                        "AwayTeam": fx["AwayTeam"],
                        "p_home": round(p[0], 4),
                        "p_draw": round(p[1], 4),
                        "p_away": round(p[2], 4),
                        "model": model_name,
                        "generated_at": generated_at,
                    }
                )

    out = pd.DataFrame(rows)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    dest = PREDICTIONS_DIR / f"{dt.date.today().isoformat()}.csv"
    if dest.exists():
        out = pd.concat([pd.read_csv(dest), out], ignore_index=True)
    out.to_csv(dest, index=False)
    return out
