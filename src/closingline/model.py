"""Dixon-Coles (1997) model with exponential time decay.

Goal means: home ~ Poisson(exp(attack_h + defense_a + home_adv)),
away ~ Poisson(exp(attack_a + defense_h)), with the Dixon-Coles tau
adjustment for low-scoring dependence. Attacks are constrained to sum
to zero for identifiability; defenses absorb the league scoring level.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Matches older than this contribute <1% weight at the default decay.
TRAIN_SEASONS = 8

# Decay rate per day; half-life ~= 1 year, in line with the Dixon-Coles
# literature's preferred range.
DEFAULT_XI = 0.0019

MAX_GOALS = 10


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    out = np.ones_like(lam)
    out = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, out)
    out = np.where((hg == 0) & (ag == 1), 1 + lam * rho, out)
    out = np.where((hg == 1) & (ag == 0), 1 + mu * rho, out)
    out = np.where((hg == 1) & (ag == 1), 1 - rho, out)
    return np.clip(out, 1e-10, None)


class DixonColes:
    def __init__(self, xi: float = DEFAULT_XI):
        self.xi = xi
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, matches: pd.DataFrame, as_of: dt.date | None = None) -> "DixonColes":
        """matches: columns HomeTeam, AwayTeam, FTHG, FTAG, Date."""
        as_of = as_of or dt.date.today()
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=365 * TRAIN_SEASONS)
        m = matches[(matches["Date"] >= cutoff) & (matches["Date"] < pd.Timestamp(as_of))]

        self.teams = sorted(set(m["HomeTeam"]) | set(m["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = m["HomeTeam"].map(idx).to_numpy()
        ai = m["AwayTeam"].map(idx).to_numpy()
        hg = m["FTHG"].to_numpy(dtype=float)
        ag = m["FTAG"].to_numpy(dtype=float)
        days = (pd.Timestamp(as_of) - m["Date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)

        # Parameter vector: attacks[0..n-2] (last = -sum), defenses[0..n-1],
        # home_adv, rho.
        def unpack(p):
            att = np.append(p[: n - 1], -p[: n - 1].sum())
            dfn = p[n - 1 : 2 * n - 1]
            return att, dfn, p[-2], p[-1]

        def nll(p):
            att, dfn, home, rho = unpack(p)
            lam = np.exp(att[hi] + dfn[ai] + home)
            mu = np.exp(att[ai] + dfn[hi])
            ll = (
                np.log(_tau(hg, ag, lam, mu, rho))
                + hg * np.log(lam) - lam
                + ag * np.log(mu) - mu
            )
            return -(w * ll).sum()

        p0 = np.concatenate([np.zeros(n - 1), np.full(n, 0.1), [0.25, -0.05]])
        res = minimize(nll, p0, method="L-BFGS-B")
        att, dfn, self.home_adv, self.rho = unpack(res.x)
        self.attack = dict(zip(self.teams, att))
        self.defense = dict(zip(self.teams, dfn))
        return self

    def _params_for(self, team: str) -> tuple[float, float]:
        if team in self.attack:
            return self.attack[team], self.defense[team]
        # Unseen team (e.g. newly promoted): proxy with the average of the
        # four weakest teams in the training window.
        ranked = sorted(self.teams, key=lambda t: self.attack[t] - self.defense[t])[:4]
        return (
            float(np.mean([self.attack[t] for t in ranked])),
            float(np.mean([self.defense[t] for t in ranked])),
        )

    def predict(self, home: str, away: str) -> tuple[float, float, float]:
        """Return (p_home, p_draw, p_away)."""
        att_h, dfn_h = self._params_for(home)
        att_a, dfn_a = self._params_for(away)
        lam = np.exp(att_h + dfn_a + self.home_adv)
        mu = np.exp(att_a + dfn_h)

        goals = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(goals, lam)
        pa = poisson.pmf(goals, mu)
        grid = np.outer(ph, pa)
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        grid *= _tau(hg, ag, np.full_like(grid, lam), np.full_like(grid, mu), self.rho)
        grid /= grid.sum()

        p_home = float(np.tril(grid, -1).sum())
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        return p_home, p_draw, p_away
