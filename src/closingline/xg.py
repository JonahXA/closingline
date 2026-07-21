"""Per-match expected goals from Understat.

Understat league pages embed a `datesData` JSON blob with both teams' xG
for every finished match, covering exactly our five leagues back to 2014.
Fetching is polite (cached seasons are never re-fetched, 1s delay) and
everything downstream treats xG as optional — if a match has no xG the
features are NaN and the GBM handles them natively.
"""

from __future__ import annotations

import difflib
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from .data import DATA_DIR, LEAGUES, _session, current_season_start_year

XG_DIR = DATA_DIR.parent / "xg"
PLAYERS_DIR = DATA_DIR.parent / "xg_players"

PLAYER_FIELDS = [
    "id", "player_name", "team_title", "position", "games", "time",
    "goals", "xG", "assists", "xA", "shots", "key_passes", "npg", "npxG",
]

UNDERSTAT_LEAGUES = {
    "E0": "EPL",
    "SP1": "La_liga",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}

FIRST_SEASON = 2014

# Hand-checked aliases where fuzzy matching fails: Understat title ->
# football-data team name.
ALIASES = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Sheffield United": "Sheffield United",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Espanyol": "Espanol",
    "Celta Vigo": "Celta",
    "Alaves": "Alaves",
    "Borussia M.Gladbach": "M'gladbach",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Hertha Berlin": "Hertha",
    "Schalke 04": "Schalke 04",
    "VfB Stuttgart": "Stuttgart",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Paris Saint Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",
    "SPAL 2013": "Spal",
    "Arminia Bielefeld": "Bielefeld",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Deportivo La Coruna": "La Coruna",
    "Sporting Gijon": "Sp Gijon",
}


def download_xg(force_current: bool = True) -> None:
    XG_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start_year()
    for div, ust in UNDERSTAT_LEAGUES.items():
        for year in range(FIRST_SEASON, current + 1):
            dest = XG_DIR / f"{ust}_{year}.csv"
            if dest.exists() and not (force_current and year >= current - 1):
                continue
            try:
                resp = _session.get(
                    f"https://understat.com/getLeagueData/{ust}/{year}",
                    timeout=30,
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "Referer": f"https://understat.com/league/{ust}/{year}",
                    },
                )
                resp.raise_for_status()
                blob = resp.json().get("dates", [])
            except Exception as e:  # noqa: BLE001 — xG is optional, never fatal
                print(f"xg fetch failed for {ust} {year}: {e}")
                continue
            rows = [
                {
                    "Div": div,
                    "Date": r["datetime"][:10],
                    "home_ust": r["h"]["title"],
                    "away_ust": r["a"]["title"],
                    "xg_h": float(r["xG"]["h"]),
                    "xg_a": float(r["xG"]["a"]),
                }
                for r in blob
                if r.get("isResult") and r.get("xG", {}).get("h") is not None
            ]
            if rows:
                pd.DataFrame(rows).to_csv(dest, index=False)
            time.sleep(1)


def download_players(force_current: bool = True) -> None:
    """Season-aggregate per-player xG/xA/minutes from the same league
    endpoint. Raw material for future player-level ratings."""
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start_year()
    for div, ust in UNDERSTAT_LEAGUES.items():
        for year in range(FIRST_SEASON, current + 1):
            dest = PLAYERS_DIR / f"{ust}_{year}.csv"
            if dest.exists() and not (force_current and year >= current - 1):
                continue
            try:
                resp = _session.get(
                    f"https://understat.com/getLeagueData/{ust}/{year}",
                    timeout=30,
                    headers={
                        "X-Requested-With": "XMLHttpRequest",
                        "Referer": f"https://understat.com/league/{ust}/{year}",
                    },
                )
                resp.raise_for_status()
                players = resp.json().get("players", [])
            except Exception as e:  # noqa: BLE001
                print(f"player fetch failed for {ust} {year}: {e}")
                continue
            if players:
                df = pd.DataFrame(players)
                keep = [c for c in PLAYER_FIELDS if c in df.columns]
                df[keep].assign(Div=div, season=year).to_csv(dest, index=False)
            time.sleep(1)


def load_players() -> pd.DataFrame:
    files = sorted(PLAYERS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(fc|cf|ac|as|ss|sc|rc|cd|ud|sd|us|afc)\b", " ", s.lower())
    return re.sub(r"[^a-z]", "", s)


def build_name_map(xg: pd.DataFrame, results: pd.DataFrame) -> dict[str, str]:
    """Map Understat titles to football-data names, per division."""
    mapping: dict[str, str] = {}
    for div in xg["Div"].unique():
        fd_teams = sorted(
            set(results.loc[results["Div"] == div, "HomeTeam"])
            | set(results.loc[results["Div"] == div, "AwayTeam"])
        )
        fd_norm = {_norm(t): t for t in fd_teams}
        for ust in sorted(set(xg.loc[xg["Div"] == div, "home_ust"])):
            if ust in ALIASES and ALIASES[ust] in fd_teams:
                mapping[ust] = ALIASES[ust]
                continue
            n = _norm(ust)
            if n in fd_norm:
                mapping[ust] = fd_norm[n]
                continue
            close = difflib.get_close_matches(n, list(fd_norm), n=1, cutoff=0.75)
            if close:
                mapping[ust] = fd_norm[close[0]]
    return mapping


def load_xg() -> pd.DataFrame:
    files = sorted(XG_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["Div", "Date", "home_ust", "away_ust", "xg_h", "xg_a"])
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def attach_xg(results: pd.DataFrame) -> pd.DataFrame:
    """Add XGH/XGA columns to a results frame (NaN where no xG exists)."""
    xg = load_xg()
    if xg.empty:
        return results.assign(XGH=float("nan"), XGA=float("nan"))
    mapping = build_name_map(xg, results)
    xg = xg.assign(
        HomeTeam=xg["home_ust"].map(mapping),
        AwayTeam=xg["away_ust"].map(mapping),
    ).dropna(subset=["HomeTeam", "AwayTeam"])

    keyed = xg.set_index(["Div", "Date", "HomeTeam", "AwayTeam"])[["xg_h", "xg_a"]]
    keyed = keyed[~keyed.index.duplicated()]
    xgh = xga = None
    # Kickoff dates can differ by a day between sources; try exact, then ±1.
    for offset in (0, 1, -1):
        dates = (results["Date"] + pd.Timedelta(days=offset)).dt.date.astype(str)
        idx = pd.MultiIndex.from_arrays(
            [results["Div"], dates, results["HomeTeam"], results["AwayTeam"]]
        )
        found = keyed.reindex(idx)
        if xgh is None:
            xgh = found["xg_h"].to_numpy().copy()
            xga = found["xg_a"].to_numpy().copy()
        else:
            fill = pd.isna(xgh)
            xgh[fill] = found["xg_h"].to_numpy()[fill]
            xga[fill] = found["xg_a"].to_numpy()[fill]
    out = results.copy()  # defragment before adding columns
    out["XGH"] = xgh
    out["XGA"] = xga
    return out
