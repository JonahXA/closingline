"""Convert bookmaker odds to de-vigged implied probabilities."""

from __future__ import annotations

import pandas as pd

# Preference order: Pinnacle closing, Bet365 closing, Pinnacle, Bet365,
# then market averages. Each entry is (home, draw, away) column names.
ODDS_SOURCES = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("AvgH", "AvgD", "AvgA"),
]


def implied_probs(row: pd.Series) -> tuple[float, float, float, str] | None:
    """Best available (p_home, p_draw, p_away, source), vig removed
    proportionally. None if the row carries no usable odds."""
    for h, d, a in ODDS_SOURCES:
        if h in row.index and pd.notna(row[h]) and pd.notna(row[d]) and pd.notna(row[a]):
            raw = [1 / row[h], 1 / row[d], 1 / row[a]]
            total = sum(raw)
            if total <= 0:
                continue
            return raw[0] / total, raw[1] / total, raw[2] / total, h
    return None


def add_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append mkt_home/mkt_draw/mkt_away/odds_source columns (NaN when absent)."""
    out = df.copy()
    cols = {"mkt_home": [], "mkt_draw": [], "mkt_away": [], "odds_source": []}
    for _, row in out.iterrows():
        probs = implied_probs(row)
        if probs is None:
            for c in cols:
                cols[c].append(None)
        else:
            cols["mkt_home"].append(probs[0])
            cols["mkt_draw"].append(probs[1])
            cols["mkt_away"].append(probs[2])
            cols["odds_source"].append(probs[3])
    for c, vals in cols.items():
        out[c] = vals
    return out
