"""Dixon-Coles fit on blended goals + xG.

Goals are a noisy sample of chance quality; xG measures the chances
directly. Fitting the same Poisson structure on a goals/xG blend should
give sharper attack/defense ratings. Where xG is missing (second
divisions, pre-2014) the blend falls back to actual goals, so coverage
gaps degrade gracefully. The Poisson log-likelihood stays valid for
non-integer "goals" — the dropped log-Gamma term is parameter-free.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .model import DixonColes

# Tuned by walk-forward sweep (see reports/sweep.csv, `closingline sweep`):
# alpha=0.5 was already optimal; xi=0.003 (faster decay than the classical
# default 0.0019) won on both Brier and log loss over 5,286 matches.
XGDC_XI = 0.003
XGDC_ALPHA = 0.5


class XgDixonColes(DixonColes):
    name = "xg-dixon-coles"

    def __init__(self, xi: float = XGDC_XI, alpha: float = XGDC_ALPHA):
        super().__init__(xi=xi)
        self.alpha = alpha  # weight on actual goals; (1 - alpha) on xG
        self._pool_len = -1
        self._blended: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame, as_of: dt.date | None = None) -> "XgDixonColes":
        if self._pool_len != len(matches):
            from .xg import attach_xg

            m = attach_xg(matches)
            self._blended = matches.assign(
                FTHG=self.alpha * m["FTHG"] + (1 - self.alpha) * m["XGH"].fillna(m["FTHG"]),
                FTAG=self.alpha * m["FTAG"] + (1 - self.alpha) * m["XGA"].fillna(m["FTAG"]),
            )
            self._pool_len = len(matches)
        return super().fit(self._blended, as_of=as_of)
