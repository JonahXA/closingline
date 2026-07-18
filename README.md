# ClosingLine

**Live, pre-registered probabilistic football forecasts, publicly benchmarked against the sportsbook closing line.**

**Dashboard: [jonahxa.github.io/closingline](https://jonahxa.github.io/closingline/)**

Every day, an automated pipeline fits Dixon-Coles models on the Big-5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), issues win/draw/loss probabilities for upcoming fixtures, and commits them to this repository **before the matches are played**. Git history is the timestamp: forecasts are frozen at first issuance and never revised, so the track record is tamper-evident and fully reproducible.

The research question, extending my [ML market-efficiency study of the 2026 World Cup](https://jonahx-dev.vercel.app): *how close can open statistical models get to the calibration of the closing line — the sharpest publicly observable probability estimate in sport?*

## How it works

1. **Ingest** — historical results and closing odds from [football-data.co.uk](https://www.football-data.co.uk), plus upcoming fixtures.
2. **Model** — a small zoo: Dixon-Coles (1997) bivariate Poisson with low-score dependence correction and exponential time decay, plus an Elo-Poisson model (margin-weighted Elo ratings mapped to expected goals), pooled in an equal-weight log-linear ensemble. All fit per league (top flight + second division) by weighted MLE; every model's forecasts are published so each builds its own track record.
3. **Predict** — daily GitHub Actions run issues forecasts for fixtures in the next 7 days and commits them to [`predictions/`](predictions/). First issuance stands; nothing is overwritten.
4. **Evaluate** — once results arrive, forecasts are scored with Brier score and log loss against de-vigged closing-line probabilities (Pinnacle closing preferred).

## Usage

```bash
pip install -e .
closingline ingest      # download data
closingline predict     # issue forecasts for the next 7 days
closingline evaluate    # score issued forecasts vs results and the market
closingline backtest    # walk-forward backtest over past seasons
closingline export      # regenerate dashboard/public/data.json
```

The dashboard is a static Next.js app in [`dashboard/`](dashboard/), rebuilt and deployed to GitHub Pages on every push.

## Backtest results (3 seasons, 5,286 matches, walk-forward)

| | Brier | Log loss |
|---|---|---|
| Dixon-Coles (primary) | 0.5874 | 0.9855 |
| Ensemble (equal-weight log-pool) | 0.5885 | 0.9873 |
| Elo-Poisson | 0.5911 | 0.9914 |
| De-vigged closing line | 0.5734 | 0.9644 |

The market wins everywhere — for now. The size of the gap is the research result, and shrinking it is the roadmap. Honest finding: the naive ensemble does **not** beat Dixon-Coles — Elo-Poisson is weaker and highly correlated, so equal-weight pooling dilutes the better model. All three models' forecasts are published daily so each builds its own live track record.

## Roadmap

- [x] v1: Dixon-Coles baseline, daily automated forecasts, Brier/log-loss evaluation
- [x] Public dashboard (Next.js + Recharts on GitHub Pages) with calibration plots and league tables
- [x] Handle promoted teams properly (train on second-division data)
- [x] Walk-forward backtest suite across past seasons
- [x] Model zoo v1: Elo-Poisson + equal-weight ensemble, all tracks published
- [ ] Weighted / stacked ensembling; gradient-boosted feature model
- [ ] Expected-value analysis vs opening vs closing lines (CLV study)

## Disclaimer

This is a statistical modeling and market-efficiency research project. It is **not betting advice**, and no wagering is involved or endorsed.

---

Built by [Jonah Alsfasser](https://jonahx-dev.vercel.app) · Data © [football-data.co.uk](https://www.football-data.co.uk)
