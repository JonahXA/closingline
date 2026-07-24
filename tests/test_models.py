"""Fast unit tests on synthetic data. The one that matters most is
causality: features and ratings for a match must never change when
future matches are added."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from closingline.features import FEATURE_COLS, build_features
from closingline.markets import implied_probs
from closingline.model import DixonColes
from closingline.zoo import combine, equal_weights, fit_pool_weights


def synthetic_matches(n_rounds: int = 40, seed: int = 0) -> pd.DataFrame:
    """Round-robin league where 'Strong' outscores 'Weak' on average."""
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Mid1", "Mid2", "Weak"]
    strength = {"Strong": 1.9, "Mid1": 1.3, "Mid2": 1.2, "Weak": 0.7}
    rows = []
    date = pd.Timestamp("2024-01-01")
    for r in range(n_rounds):
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                home, away = (h, a) if r % 2 == 0 else (a, h)
                rows.append(
                    {
                        "Div": "T1",
                        "Date": date,
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": rng.poisson(strength[home] * 1.2),
                        "FTAG": rng.poisson(strength[away]),
                        "HST": rng.poisson(strength[home] * 4),
                        "AST": rng.poisson(strength[away] * 4),
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_features_are_causal():
    m = synthetic_matches()
    half = len(m) // 2
    feats_full, _ = build_features(m)
    feats_half, _ = build_features(m.iloc[:half])
    pd.testing.assert_frame_equal(
        feats_full.iloc[:half][FEATURE_COLS], feats_half[FEATURE_COLS]
    )


def test_dixon_coles_learns_strength():
    m = synthetic_matches()
    model = DixonColes().fit(m, as_of=m["Date"].max().date() + dt.timedelta(days=1))
    p_home, p_draw, p_away = model.predict("Strong", "Weak")
    assert p_home > 0.5 > p_away
    assert abs(p_home + p_draw + p_away - 1) < 1e-9


def test_dixon_coles_unseen_team_fallback():
    m = synthetic_matches()
    model = DixonColes().fit(m, as_of=m["Date"].max().date() + dt.timedelta(days=1))
    p = model.predict("Strong", "Newcomer")
    assert abs(sum(p) - 1) < 1e-9
    assert p[0] > 0.4  # strong side still favored over an unknown


def test_combine_normalizes_and_matches_geometric_mean():
    probs = [(0.5, 0.3, 0.2), (0.7, 0.2, 0.1)]
    p = combine(probs, equal_weights(2))
    assert abs(sum(p) - 1) < 1e-9
    geo = np.sqrt(np.array(probs[0]) * np.array(probs[1]))
    geo /= geo.sum()
    assert np.allclose(p, geo)


def test_pool_weights_fallback_when_insufficient():
    w = fit_pool_weights(pd.DataFrame(columns=["model"]), ["a", "b"])
    assert np.allclose(w, [0.5, 0.5])


def test_pool_weights_prefer_better_model():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(600):
        outcome = rng.integers(0, 3)
        good = np.full(3, 0.15)
        good[outcome] = 0.7
        bad = np.full(3, 1 / 3)
        for name, p in [("good", good), ("bad", bad)]:
            rows.append(
                {
                    "model": name,
                    "Div": "T1",
                    "Date": f"2025-01-{i % 28 + 1:02d}",
                    "HomeTeam": f"H{i}",
                    "AwayTeam": f"A{i}",
                    "FTHG": 1 if outcome == 1 else (2 if outcome == 0 else 0),
                    "FTAG": 1 if outcome == 1 else (0 if outcome == 0 else 2),
                    "p_home": p[0],
                    "p_draw": p[1],
                    "p_away": p[2],
                }
            )
    w = fit_pool_weights(pd.DataFrame(rows), ["good", "bad"])
    assert w[0] > 0.8


def test_xgdc_falls_back_to_goals_without_xg_data():
    from closingline.xgdc import XgDixonColes

    m = synthetic_matches()
    as_of = m["Date"].max().date() + dt.timedelta(days=1)
    # Match decay so only the goals/xG blend can differ; synthetic teams
    # have no xG, so the fallback must reproduce plain Dixon-Coles exactly.
    dc = DixonColes(xi=XgDixonColes().xi).fit(m, as_of=as_of)
    xgdc = XgDixonColes().fit(m, as_of=as_of)
    assert np.allclose(dc.predict("Strong", "Weak"), xgdc.predict("Strong", "Weak"), atol=1e-6)


def test_shrinkage_pulls_ratings_toward_the_mean():
    """Empirical-Bayes shrinkage must compress the spread of team ratings,
    and must reduce to plain MLE when the penalty is zero."""
    m = synthetic_matches()
    as_of = m["Date"].max().date() + dt.timedelta(days=1)

    plain = DixonColes().fit(m, as_of=as_of)
    shrunk = DixonColes(shrinkage=5.0).fit(m, as_of=as_of)

    spread_plain = np.std(list(plain.attack.values()))
    spread_shrunk = np.std(list(shrunk.attack.values()))
    assert spread_shrunk < spread_plain

    # shrinkage=0 must be exactly the unpenalized fit.
    zero = DixonColes(shrinkage=0.0).fit(m, as_of=as_of)
    assert np.allclose(
        zero.predict("Strong", "Weak"), plain.predict("Strong", "Weak"), atol=1e-9
    )


def test_squad_features_use_only_prior_season_production():
    """The leak this guards: Understat player stats are season aggregates,
    so a season's own production must never inform that season's rating."""
    from closingline.squad import team_features

    rows = []
    for season, xg in [(2020, 1.0), (2021, 50.0)]:  # implausible 2021 spike
        for pid in range(4):
            rows.append(
                {
                    "id": pid, "player_name": f"p{pid}", "team_title": "Alpha",
                    "position": "F", "games": 30, "time": 2700, "goals": 10,
                    "xG": xg, "assists": 2, "xA": 0.0, "shots": 40,
                    "key_passes": 10, "npg": 9, "npxG": xg, "Div": "E0", "season": season,
                }
            )
    feats = team_features(pd.DataFrame(rows))
    row2021 = feats[(feats["season"] == 2021) & (feats["team_title"] == "Alpha")].iloc[0]
    # 2021's rating must reflect 2020's weak output, not 2021's spike.
    per90_2020 = 1.0 * 90 / 2700
    # squad_strength is stored rounded to 5dp.
    assert row2021["squad_strength"] == pytest.approx(per90_2020, abs=1e-5)


def test_squad_splits_multi_team_transfer_rows():
    from closingline.squad import _split_transfers

    df = pd.DataFrame([{"team_title": "Fulham,West Ham", "time": 1000, "id": 1}])
    out = _split_transfers(df)
    assert set(out["team_title"]) == {"Fulham", "West Ham"}
    assert out["time"].tolist() == [500.0, 500.0]


def test_significance_detects_real_gap_and_ignores_noise():
    from closingline.significance import _bootstrap, _diebold_mariano

    rng = np.random.default_rng(0)
    # A true small positive differential (signal) must register on both tests.
    signal = rng.normal(0.01, 0.05, size=4000)
    lo, hi, p_boot = _bootstrap(signal)
    dm, p_dm = _diebold_mariano(signal)
    assert lo > 0 and p_boot < 0.05 and p_dm < 0.05

    # Zero-mean noise must not.
    noise = rng.normal(0.0, 0.05, size=4000)
    lo, hi, p_boot = _bootstrap(noise)
    _, p_dm = _diebold_mariano(noise)
    assert lo < 0 < hi and p_boot > 0.05 and p_dm > 0.05


def test_implied_probs_devig_and_source_preference():
    row = pd.Series({"PSCH": 2.0, "PSCD": 3.5, "PSCA": 4.0, "B365H": 1.9, "B365D": 3.4, "B365A": 3.9})
    p_home, p_draw, p_away, source = implied_probs(row)
    assert source == "PSCH"
    assert abs(p_home + p_draw + p_away - 1) < 1e-9
    assert p_home > p_draw > p_away
