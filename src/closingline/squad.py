"""Squad-strength features from per-player xG.

The bias scan showed the market accumulating in-season information faster
than our rating models. Squad quality is a prime candidate: to Dixon-Coles,
a team that sold its best scorer looks identical to last season's team
until enough new matches accrue.

LEAKAGE DISCIPLINE — the reason this module is careful: Understat gives
*season-aggregate* player stats, so a player's 2025 xG total includes
matches after any mid-2025 prediction date. Using the current season's
totals would leak the future. So every feature here is built strictly from
**completed prior seasons**:

  * player quality = prior-season npxG+xA per 90, credible-weighted by
    minutes (a 200-minute cameo is not evidence of a 0.9/90 striker),
  * squad strength = minutes-weighted mean quality of players who appear
    for the team in the season being predicted,
  * continuity = share of the prior squad's minutes retained.

Squad membership for the current season is itself known before kickoff
(transfer windows close), so using *who is in the squad* is fair; only
their *production this season* would be a leak, and that is never read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .xg import build_name_map, load_players

# Minutes of prior-season evidence for a player's rate to count fully.
# Below this, the rate is shrunk toward the league-average player.
CREDIBILITY_MINUTES = 900.0

MIN_SEASON = 2015  # first season with a prior season to draw on


def _per90(df: pd.DataFrame) -> pd.Series:
    minutes = df["time"].clip(lower=1)
    return (df["npxG"].fillna(0) + df["xA"].fillna(0)) * 90.0 / minutes


def player_quality(players: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: credibility-weighted attacking output per 90.

    Returns one row per (id, season) where `season` is the season the
    production came FROM — callers must shift to use it as a prior.
    """
    df = players.copy()
    df["raw_p90"] = _per90(df)
    league_mean = df.groupby("season")["raw_p90"].transform("mean")
    # Credibility weighting: z = min(1, minutes / CREDIBILITY_MINUTES).
    z = (df["time"] / CREDIBILITY_MINUTES).clip(upper=1.0)
    df["quality"] = z * df["raw_p90"] + (1 - z) * league_mean
    return df[["id", "player_name", "team_title", "season", "time", "quality"]]


def _split_transfers(players: pd.DataFrame) -> pd.DataFrame:
    """Understat reports mid-season transfers as one row with a
    comma-joined team string ("Fulham,West Ham", ~3% of rows). Expand to
    one row per club, splitting minutes evenly — the aggregate doesn't say
    how they were divided, and crediting either club in full is worse.
    """
    df = players.copy()
    df["team_title"] = df["team_title"].fillna("").str.split(",")
    n_clubs = df["team_title"].apply(len).clip(lower=1)
    df["time"] = df["time"] / n_clubs
    df = df.explode("team_title")
    df["team_title"] = df["team_title"].str.strip()
    return df[df["team_title"] != ""]


def team_features(players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (Div, season, team_title): squad strength and continuity, using
    only prior-season production."""
    players = load_players() if players is None else players
    if players.empty:
        return pd.DataFrame()
    players = _split_transfers(players)

    q = player_quality(players)
    # Prior-season quality lookup, indexed by (player id, season it applies to).
    prior = q[["id", "season", "quality"]].copy()
    prior["season"] = prior["season"] + 1  # applies to the following season
    prior = prior.drop_duplicates(subset=["id", "season"], keep="last")

    # Squad membership + minutes for the season being predicted.
    squads = players[["Div", "season", "team_title", "id", "time"]].copy()
    squads = squads.merge(prior, on=["id", "season"], how="left")

    rows = []
    for (div, season, team), g in squads.groupby(["Div", "season", "team_title"]):
        if season < MIN_SEASON:
            continue
        known = g.dropna(subset=["quality"])
        if known.empty:
            continue
        w = known["time"].clip(lower=1)
        strength = float((known["quality"] * w).sum() / w.sum())
        # Continuity: share of THIS squad's minutes played by players who
        # have prior-season evidence at all (i.e. not brand-new to the data).
        continuity = float(known["time"].sum() / max(g["time"].sum(), 1))
        rows.append(
            {
                "Div": div,
                "season": int(season),
                "team_title": team,
                "squad_strength": round(strength, 5),
                "squad_continuity": round(continuity, 4),
                "squad_players_known": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def attach_squad(results: pd.DataFrame) -> pd.DataFrame:
    """Add home/away squad_strength and squad_continuity to a results frame.

    Season is derived from the match date (July cutoff), and names are
    mapped through the same Understat->football-data table used for xG.
    Missing values stay NaN — the GBM handles them natively.
    """
    feats = team_features()
    if feats.empty:
        return results.assign(
            h_squad=np.nan, a_squad=np.nan, h_continuity=np.nan, a_continuity=np.nan
        )

    players = _split_transfers(load_players())
    xg_like = players.rename(columns={"team_title": "home_ust"})[["Div", "home_ust"]]
    mapping = build_name_map(xg_like.assign(Date="", away_ust="", xg_h=0, xg_a=0), results)
    feats["team"] = feats["team_title"].map(mapping)
    feats = feats.dropna(subset=["team"])

    season = np.where(results["Date"].dt.month >= 7, results["Date"].dt.year,
                      results["Date"].dt.year - 1)
    base = results.assign(_season=season)

    keyed = feats.set_index(["Div", "season", "team"])[["squad_strength", "squad_continuity"]]
    keyed = keyed[~keyed.index.duplicated()]

    def lookup(team_col: str) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays([base["Div"], base["_season"], base[team_col]])
        return keyed.reindex(idx)

    h = lookup("HomeTeam")
    a = lookup("AwayTeam")
    out = results.copy()
    out["h_squad"] = h["squad_strength"].to_numpy()
    out["a_squad"] = a["squad_strength"].to_numpy()
    out["h_continuity"] = h["squad_continuity"].to_numpy()
    out["a_continuity"] = a["squad_continuity"].to_numpy()
    return out
