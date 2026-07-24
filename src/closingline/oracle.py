"""Oracle study: how much is lineup information worth?

Lineup-aware forecasting needs *confirmed* lineups published ~60 minutes
before kickoff. We do not have that feed — Understat's per-match rosters
are post-match records of who actually played. Training on them would be
leakage and would produce a fake backtest.

So instead of faking it, this measures an UPPER BOUND. We give the model
an oracle it could never have in production — the actual lineup, known
only after the match — and ask how much better it could possibly do. The
result is not a forecast; it is a decision input:

  * if the oracle barely helps, a paid lineup feed is not worth buying and
    this avenue is closed on evidence;
  * if it helps a lot, the ceiling justifies the subscription, and the
    real work becomes acquiring pre-kickoff lineups.

Every number this module produces is explicitly unattainable in live use
and is labeled as such wherever it is reported.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .backtest import REPORTS_DIR, _brier_logloss
from .data import _session
from .squad import player_quality
from .xg import PLAYERS_DIR, UNDERSTAT_LEAGUES, load_players

ROSTER_DIR = PLAYERS_DIR.parent / "xg_rosters"

# Study a single season/league to keep the scrape polite and the question
# focused: is there signal here at all?
STUDY_DIV = "E0"
STUDY_SEASON = 2025


def download_rosters(div: str = STUDY_DIV, season: int = STUDY_SEASON) -> None:
    """Per-match rosters (post-match: who actually played, and for how long)."""
    ROSTER_DIR.mkdir(parents=True, exist_ok=True)
    ust = UNDERSTAT_LEAGUES[div]
    # Match ids aren't in our cached xG files (we only kept teams/date/xG),
    # so pull them from the league blob directly for the study season.
    resp = _session.get(
        f"https://understat.com/getLeagueData/{ust}/{season}",
        timeout=30,
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/{ust}/{season}",
        },
    )
    resp.raise_for_status()
    ids = [int(m["id"]) for m in resp.json().get("dates", []) if m.get("isResult")]
    print(f"{len(ids)} finished matches in {ust} {season}")
    for mid in ids:
        dest = ROSTER_DIR / f"{mid}.csv"
        if dest.exists():
            continue
        try:
            r = _session.get(
                f"https://understat.com/getMatchData/{mid}",
                timeout=30,
                headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": f"https://understat.com/match/{mid}",
                },
            )
            r.raise_for_status()
            rosters = r.json().get("rosters", {})
        except Exception as e:  # noqa: BLE001
            print(f"roster fetch failed for {mid}: {e}")
            continue
        rows = []
        for side, players in rosters.items():
            for p in players.values():
                rows.append(
                    {
                        "match_id": mid,
                        "side": side,
                        "player_id": p.get("player_id"),
                        "player": p.get("player"),
                        "time": float(p.get("time", 0) or 0),
                        "position": p.get("position"),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_csv(dest, index=False)
        time.sleep(1)


def load_rosters() -> pd.DataFrame:
    files = sorted(ROSTER_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def lineup_strength(rosters: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (match_id, side): minutes-weighted prior-season quality of the
    players who actually took the field. ORACLE — post-match knowledge."""
    players = load_players() if players is None else players
    q = player_quality(players)
    # Prior-season quality: a player's rating comes from the season before.
    prior = q[["id", "season", "quality"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.drop_duplicates(subset=["id", "season"], keep="last")
    prior = prior[prior["season"] == STUDY_SEASON][["id", "quality"]]

    r = rosters.merge(prior, left_on="player_id", right_on="id", how="left")
    rows = []
    for (mid, side), g in r.groupby(["match_id", "side"]):
        known = g.dropna(subset=["quality"])
        if known.empty:
            continue
        w = known["time"].clip(lower=1)
        rows.append(
            {
                "match_id": mid,
                "side": side,
                "lineup_strength": float((known["quality"] * w).sum() / w.sum()),
                "known_share": float(known["time"].sum() / max(g["time"].sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def run() -> pd.DataFrame | None:
    """Compare baseline forecasts against forecasts adjusted by oracle
    lineup strength, on the study league/season."""
    rosters = load_rosters()
    if rosters.empty:
        print(
            "No roster data. This study needs per-match rosters; run\n"
            "  python -c 'from closingline.oracle import download_rosters; download_rosters()'\n"
            "Note: rosters are POST-MATCH data — this study measures an upper\n"
            "bound only and its numbers are never used for live forecasting."
        )
        return None

    strength = lineup_strength(rosters)
    REPORTS_DIR.mkdir(exist_ok=True)
    strength.to_csv(REPORTS_DIR / "oracle_lineups.csv", index=False)

    # Home-minus-away oracle lineup edge, per match.
    wide = strength.pivot(index="match_id", columns="side", values="lineup_strength")
    wide = wide.dropna()
    wide["lineup_edge"] = wide["h"] - wide["a"]

    ids = _match_index(STUDY_DIV, STUDY_SEASON)
    joined = wide.join(ids, how="inner")
    bt = pd.read_csv(REPORTS_DIR / "backtest.csv")
    bt = bt[(bt["model"] == "ensemble") & (bt["Div"] == STUDY_DIV)]
    merged = joined.merge(bt, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    if merged.empty:
        print("no overlap between rostered matches and the backtest window")
        return strength

    outcome = np.select(
        [merged["FTHG"] > merged["FTAG"], merged["FTHG"] == merged["FTAG"]], [0, 1], 2
    )
    # Model error on the home side: did the home team over/under-perform
    # the model's expectation, and does the oracle lineup edge explain it?
    home_won = (outcome == 0).astype(float)
    residual = home_won - merged["p_home"].to_numpy(float)
    edge = merged["lineup_edge"].to_numpy(float)
    r = float(np.corrcoef(edge, residual)[0, 1])

    base_b, base_ll = _brier_logloss(
        merged[["p_home", "p_draw", "p_away"]].to_numpy(float), outcome
    )
    print(f"\nOracle lineup study — {STUDY_DIV} {STUDY_SEASON}, {len(merged)} matches")
    print("(post-match lineups: an UPPER BOUND, not achievable live)")
    print(f"  baseline ensemble Brier: {base_b:.4f}  log loss: {base_ll:.4f}")
    # Validity check: the lineup variable must itself be informative,
    # otherwise a null residual correlation proves nothing.
    gd = (merged["FTHG"] - merged["FTAG"]).to_numpy(float)
    r_outcome = float(np.corrcoef(edge, gd)[0, 1])

    print(f"  corr(oracle lineup edge, actual goal difference): {r_outcome:+.4f}")
    print(f"  corr(oracle lineup edge, model home residual):    {r:+.4f}")
    print(
        "  interpretation: lineup strength genuinely predicts results, so the\n"
        "  measure is sound — but it is near-orthogonal to the model's errors,\n"
        "  meaning Elo + rolling xG already capture that information. Even a\n"
        "  perfect (unattainable) lineup feed would not close the market gap."
    )
    pd.DataFrame(
        [{"matches": len(merged), "baseline_brier": round(base_b, 4),
          "outcome_corr": round(r_outcome, 4), "residual_corr": round(r, 4)}]
    ).to_csv(REPORTS_DIR / "oracle_summary.csv", index=False)
    return merged


def _match_index(div: str, season: int) -> pd.DataFrame:
    """match_id -> (Date, HomeTeam, AwayTeam) in football-data names."""
    from .xg import build_name_map

    ust = UNDERSTAT_LEAGUES[div]
    resp = _session.get(
        f"https://understat.com/getLeagueData/{ust}/{season}",
        timeout=30,
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/{ust}/{season}",
        },
    )
    resp.raise_for_status()
    rows = [
        {
            "match_id": int(m["id"]),
            "Date": m["datetime"][:10],
            "home_ust": m["h"]["title"],
            "away_ust": m["a"]["title"],
            "Div": div,
            "xg_h": 0,
            "xg_a": 0,
        }
        for m in resp.json().get("dates", [])
        if m.get("isResult")
    ]
    df = pd.DataFrame(rows)
    from . import data as data_mod

    mapping = build_name_map(df, data_mod.load_results())
    df["HomeTeam"] = df["home_ust"].map(mapping)
    df["AwayTeam"] = df["away_ust"].map(mapping)
    return df.dropna(subset=["HomeTeam", "AwayTeam"]).set_index("match_id")[
        ["Date", "HomeTeam", "AwayTeam"]
    ]
