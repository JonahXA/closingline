"""Elo-Poisson model.

Sequential Elo ratings (goal-margin weighted) over the full match history,
then two recency-weighted Poisson regressions map the pre-match rating
difference to expected home and away goals. Structurally different from
Dixon-Coles — ratings update match-by-match instead of being refit — which
is what makes it worth ensembling.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .model import DEFAULT_XI, MAX_GOALS

START_RATING = 1500.0


class EloPoisson:
    name = "elo-poisson"

    def __init__(self, k: float = 20.0, hfa: float = 60.0, xi: float = DEFAULT_XI):
        self.k = k
        self.hfa = hfa
        self.xi = xi
        self.ratings: dict[str, float] = {}
        self.coef_home = np.array([0.2, 0.8])
        self.coef_away = np.array([0.05, -0.8])

    def fit(self, matches: pd.DataFrame, as_of: dt.date | None = None) -> "EloPoisson":
        as_of = as_of or dt.date.today()
        m = matches[matches["Date"] < pd.Timestamp(as_of)]

        ratings: dict[str, float] = {}
        diffs = np.empty(len(m))
        for i, (h, a, hg, ag) in enumerate(
            zip(m["HomeTeam"], m["AwayTeam"], m["FTHG"], m["FTAG"])
        ):
            rh = ratings.get(h, START_RATING)
            ra = ratings.get(a, START_RATING)
            diffs[i] = (rh - ra) / 400.0
            expected = 1.0 / (1.0 + 10 ** (-(rh + self.hfa - ra) / 400.0))
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = self.k * np.log1p(abs(hg - ag)) * (score - expected)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
        self.ratings = ratings

        days = (pd.Timestamp(as_of) - m["Date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)
        hg = m["FTHG"].to_numpy(dtype=float)
        ag = m["FTAG"].to_numpy(dtype=float)

        def fit_glm(goals: np.ndarray, init: np.ndarray) -> np.ndarray:
            def nll(p):
                lam = np.exp(p[0] + p[1] * diffs)
                return -(w * (goals * np.log(lam) - lam)).sum()

            return minimize(nll, init, method="L-BFGS-B").x

        self.coef_home = fit_glm(hg, self.coef_home)
        self.coef_away = fit_glm(ag, self.coef_away)
        return self

    def _rating_for(self, team: str) -> float:
        if team in self.ratings:
            return self.ratings[team]
        ranked = sorted(self.ratings.values())[:4]
        return float(np.mean(ranked)) if ranked else START_RATING

    def predict(self, home: str, away: str) -> tuple[float, float, float]:
        d = (self._rating_for(home) - self._rating_for(away)) / 400.0
        lam = np.exp(self.coef_home[0] + self.coef_home[1] * d)
        mu = np.exp(self.coef_away[0] + self.coef_away[1] * d)

        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        grid /= grid.sum()
        return (
            float(np.tril(grid, -1).sum()),
            float(np.trace(grid)),
            float(np.triu(grid, 1).sum()),
        )
