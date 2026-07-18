"""Model registry and the ensemble.

Every model implements fit(matches, as_of) -> self and
predict(home, away) -> (p_home, p_draw, p_away).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .elo import EloPoisson
from .model import DixonColes

# The model used for headline stats, live tables, and single-model charts.
# Chosen by backtest: Dixon-Coles beat both Elo-Poisson and the equal-weight
# ensemble over 3 seasons (Brier 0.5874 vs 0.5911 / 0.5885).
PRIMARY_MODEL = "dixon-coles"


class Ensemble:
    """Log-linear pool of the component models, equal weights."""

    name = "ensemble"

    def __init__(self) -> None:
        self.components = [DixonColes(), EloPoisson()]

    def fit(self, matches: pd.DataFrame, as_of: dt.date | None = None) -> "Ensemble":
        for c in self.components:
            c.fit(matches, as_of=as_of)
        return self

    def predict(self, home: str, away: str) -> tuple[float, float, float]:
        logs = np.zeros(3)
        for c in self.components:
            logs += np.log(np.clip(c.predict(home, away), 1e-12, None))
        p = np.exp(logs / len(self.components))
        p /= p.sum()
        return float(p[0]), float(p[1]), float(p[2])


def build_models() -> list:
    """One fitted-per-league instance of each model, ensemble reusing the
    same fitted components so nothing is fit twice."""
    ens = Ensemble()
    dc, elo = ens.components
    return [dc, elo, ens]


def fit_all(models: list, matches: pd.DataFrame, as_of: dt.date | None = None) -> None:
    """Fit every registered model. The ensemble's components are the other
    registry entries, so fitting them fits it."""
    for m in models:
        if not isinstance(m, Ensemble):
            m.fit(matches, as_of=as_of)
