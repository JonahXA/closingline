"""Gradient-boosted outcome model.

HistGradientBoostingClassifier over causal form features (rolling goals,
points per game, rest days, Elo diff). Brings information the ratings
models ignore — recent form and schedule — which is what gives it a shot
at decorrelated errors worth ensembling.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .elo import elo_pass
from .features import FEATURE_COLS, build_features, snapshot


# Tuned by walk-forward sweep (reports/sweep_gbm.csv, `closingline sweep
# --model gbm`): 0.5931 -> 0.5902. The intuition-set defaults were slightly
# overfitting — slower learning, shallower trees, and stronger L2 all helped.
GBM_PARAMS = dict(
    max_iter=300,
    learning_rate=0.03,
    max_leaf_nodes=7,
    min_samples_leaf=40,
    l2_regularization=5.0,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=0,
)


class GradientBoosted:
    name = "gbm"

    def __init__(self, **params) -> None:
        self.params = {**GBM_PARAMS, **params}
        self._pool_len = -1
        self._features: pd.DataFrame | None = None
        self._hist = None
        self._elo_timeline = None
        self._clf: HistGradientBoostingClassifier | None = None
        self._as_of: pd.Timestamp | None = None
        self._outcome: np.ndarray | None = None

    def _build_cache(self, matches: pd.DataFrame) -> None:
        from .xg import attach_xg

        matches = attach_xg(matches)
        self._features, self._hist = build_features(matches)
        _, _, self._elo_timeline = elo_pass(matches)
        self._outcome = np.select(
            [matches["FTHG"] > matches["FTAG"], matches["FTHG"] == matches["FTAG"]], [0, 1], 2
        )
        self._pool_len = len(matches)

    def fit(self, matches: pd.DataFrame, as_of: dt.date | None = None) -> "GradientBoosted":
        """`matches` is the same pool across walk-forward windows, so the
        (causal) feature table is built once and sliced per window."""
        as_of_ts = pd.Timestamp(as_of or dt.date.today())
        if self._pool_len != len(matches):
            self._build_cache(matches)
        mask = (self._features["Date"] < as_of_ts).to_numpy()
        X = self._features.loc[mask, FEATURE_COLS].to_numpy(dtype=float)
        y = self._outcome[mask]
        self._clf = HistGradientBoostingClassifier(**self.params)
        self._clf.fit(X, y)
        self._as_of = as_of_ts
        return self

    def predict(self, home: str, away: str) -> tuple[float, float, float]:
        x = snapshot(self._hist, self._elo_timeline, home, away, self._as_of)
        p = self._clf.predict_proba(x.reshape(1, -1))[0]
        # classes_ are [0, 1, 2] = home, draw, away by construction
        return float(p[0]), float(p[1]), float(p[2])
