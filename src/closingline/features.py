"""Causal per-match features for the gradient-boosted model.

Every feature for a match is computed strictly from matches played before
it, so a feature table built over the full history is leak-free for any
walk-forward split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import elo_pass

FEATURE_COLS = [
    "elo_diff",
    "h_gf5", "h_ga5", "h_ppg5", "h_gf10", "h_ga10", "h_ppg10", "h_rest",
    "a_gf5", "a_ga5", "a_ppg5", "a_gf10", "a_ga10", "a_ppg10", "a_rest",
]

# A team record: (date, goals_for, goals_against, points).
History = dict[str, list[tuple[pd.Timestamp, int, int, int]]]


def _stats(hist: list[tuple[pd.Timestamp, int, int, int]], as_of: pd.Timestamp) -> list[float]:
    prior = [r for r in hist if r[0] < as_of]
    out: list[float] = []
    for n in (5, 10):
        recent = prior[-n:]
        if len(recent) < n:
            out += [np.nan, np.nan, np.nan]
        else:
            out += [
                sum(r[1] for r in recent) / n,
                sum(r[2] for r in recent) / n,
                sum(r[3] for r in recent) / n,
            ]
    rest = (as_of - prior[-1][0]).days if prior else np.nan
    out.append(min(rest, 60) if not np.isnan(rest) else np.nan)
    return out


def build_features(m: pd.DataFrame) -> tuple[pd.DataFrame, History]:
    """Feature row per match of `m` (chronological), plus full team history
    for snapshot features at prediction time."""
    diffs, _, _ = elo_pass(m)
    hist: History = {}
    rows = []
    for i, (date, h, a, hg, ag) in enumerate(
        zip(m["Date"], m["HomeTeam"], m["AwayTeam"], m["FTHG"], m["FTAG"])
    ):
        rows.append(
            [diffs[i]] + _stats(hist.get(h, []), date) + _stats(hist.get(a, []), date)
        )
        hp = 3 if hg > ag else 1 if hg == ag else 0
        hist.setdefault(h, []).append((date, hg, ag, hp))
        hist.setdefault(a, []).append((date, ag, hg, 3 - hp if hp != 1 else 1))
    feats = pd.DataFrame(rows, columns=FEATURE_COLS, index=m.index)
    feats["Date"] = m["Date"].to_numpy()
    return feats, hist


def snapshot(
    hist: History,
    elo_timeline: dict[str, list[tuple[pd.Timestamp, float]]],
    home: str,
    away: str,
    as_of: pd.Timestamp,
) -> np.ndarray:
    """Feature vector for an upcoming match, using state strictly before as_of."""

    def rating(team: str) -> float:
        tl = [r for d, r in elo_timeline.get(team, []) if d < as_of]
        return tl[-1] if tl else 1400.0  # unseen team: below-average prior

    elo_diff = (rating(home) - rating(away)) / 400.0
    return np.array(
        [elo_diff] + _stats(hist.get(home, []), as_of) + _stats(hist.get(away, []), as_of)
    )
