"""Causal per-match features for the gradient-boosted model.

Every feature for a match is computed strictly from matches played before
it, so a feature table built over the full history is leak-free for any
walk-forward split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import elo_pass

_SIDES = ("h", "a")
_ROLL = [
    "gf5", "ga5", "ppg5", "gf10", "ga10", "ppg10",
    "sotf5", "sota5", "sotf10", "sota10", "conv10",
    "xgf5", "xga5", "xgf10", "xga10", "rest",
]
FEATURE_COLS = ["elo_diff"] + [f"{s}_{c}" for s in _SIDES for c in _ROLL]

# A team record:
# (date, goals_for, goals_against, points, sot_for, sot_against, xg_for, xg_against).
History = dict[str, list[tuple[pd.Timestamp, int, int, int, float, float, float, float]]]


def _mean(vals: list[float]) -> float:
    clean = [v for v in vals if not np.isnan(v)]
    return sum(clean) / len(clean) if clean else np.nan


def _stats(hist: list[tuple], as_of: pd.Timestamp) -> list[float]:
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
    for n in (5, 10):
        recent = prior[-n:]
        if len(recent) < n:
            out += [np.nan, np.nan]
        else:
            out += [_mean([r[4] for r in recent]), _mean([r[5] for r in recent])]
    # Finishing quality: goals per shot on target over the last 10.
    recent = prior[-10:]
    if len(recent) < 10:
        out.append(np.nan)
    else:
        sot = _mean([r[4] for r in recent])
        out.append(sum(r[1] for r in recent) / 10 / sot if sot and sot > 0 else np.nan)
    for n in (5, 10):
        recent = prior[-n:]
        if len(recent) < n:
            out += [np.nan, np.nan]
        else:
            out += [_mean([r[6] for r in recent]), _mean([r[7] for r in recent])]
    rest = (as_of - prior[-1][0]).days if prior else np.nan
    out.append(min(rest, 60) if not np.isnan(rest) else np.nan)
    return out


def build_features(m: pd.DataFrame) -> tuple[pd.DataFrame, History]:
    """Feature row per match of `m` (chronological), plus full team history
    for snapshot features at prediction time."""
    diffs, _, _ = elo_pass(m)
    nan_col = pd.Series(np.nan, index=m.index)
    hst = m["HST"] if "HST" in m.columns else nan_col
    ast = m["AST"] if "AST" in m.columns else nan_col
    xgh = m["XGH"] if "XGH" in m.columns else nan_col
    xga = m["XGA"] if "XGA" in m.columns else nan_col
    hist: History = {}
    rows = []
    for i, (date, h, a, hg, ag, sh, sa, xh, xa) in enumerate(
        zip(m["Date"], m["HomeTeam"], m["AwayTeam"], m["FTHG"], m["FTAG"], hst, ast, xgh, xga)
    ):
        rows.append(
            [diffs[i]] + _stats(hist.get(h, []), date) + _stats(hist.get(a, []), date)
        )
        hp = 3 if hg > ag else 1 if hg == ag else 0
        hist.setdefault(h, []).append((date, hg, ag, hp, sh, sa, xh, xa))
        hist.setdefault(a, []).append((date, ag, hg, (3 - hp) if hp != 1 else 1, sa, sh, xa, xh))
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
