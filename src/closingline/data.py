"""Download and load match data from football-data.co.uk."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503])),
)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Second tiers, used only for training so promoted teams carry real
# parameters into the top flight. Keyed by the top division they feed.
SECOND_TIERS = {"E0": "E1", "SP1": "SP2", "I1": "I2", "D1": "D2", "F1": "F2"}

ALL_DIVISIONS = list(LEAGUES) + list(SECOND_TIERS.values())


def training_divisions(league: str) -> list[str]:
    """Divisions whose matches inform the model for a top-flight league."""
    return [league, SECOND_TIERS[league]]

DATA_DIR = Path("data/raw")

# Seasons of history to keep on disk. Model training uses a shorter,
# recency-weighted window (see model.TRAIN_SEASONS).
ARCHIVE_SEASONS = 12


def current_season_start_year(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_history(seasons: int = ARCHIVE_SEASONS) -> None:
    """Fetch per-season CSVs. Past seasons are cached; the two most recent
    are always re-downloaded since results accrue weekly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start_year()
    for start_year in range(current - seasons + 1, current + 1):
        code = season_code(start_year)
        for div in ALL_DIVISIONS:
            dest = DATA_DIR / f"{code}_{div}.csv"
            if dest.exists() and start_year < current - 1:
                continue
            url = f"{BASE_URL}/{code}/{div}.csv"
            try:
                resp = _session.get(url, timeout=30)
            except requests.ConnectionError:
                time.sleep(5)  # be polite after a reset, then retry once
                resp = _session.get(url, timeout=30)
            if resp.status_code == 404 or len(resp.content) < 100:
                continue
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            time.sleep(0.25)


def download_fixtures() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "fixtures.csv"
    resp = _session.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        df.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in df.columns]
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df = df[df["Div"].isin(ALL_DIVISIONS)]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.dropna(subset=["Date"])


def load_results() -> pd.DataFrame:
    """All completed matches on disk, one row per match."""
    frames = []
    for path in sorted(DATA_DIR.glob("[0-9]*_*.csv")):
        df = _read_csv(path)
        df = df.dropna(subset=["FTHG", "FTAG"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No match data found — run `closingline ingest` first.")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)
    return out.sort_values("Date").reset_index(drop=True)


def load_fixtures() -> pd.DataFrame:
    """Upcoming fixtures in the covered leagues (may be empty off-season)."""
    path = DATA_DIR / "fixtures.csv"
    if not path.exists():
        download_fixtures()
    df = _read_csv(path)
    return df[df["Div"].isin(LEAGUES)]
