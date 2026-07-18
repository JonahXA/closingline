# ClosingLine

**Live, pre-registered probabilistic football forecasts, publicly benchmarked against the sportsbook closing line.**

Every day, an automated pipeline fits Dixon-Coles models on the Big-5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), issues win/draw/loss probabilities for upcoming fixtures, and commits them to this repository **before the matches are played**. Git history is the timestamp: forecasts are frozen at first issuance and never revised, so the track record is tamper-evident and fully reproducible.

The research question, extending my [ML market-efficiency study of the 2026 World Cup](https://jonahx-dev.vercel.app): *how close can open statistical models get to the calibration of the closing line — the sharpest publicly observable probability estimate in sport?*

## How it works

1. **Ingest** — historical results and closing odds from [football-data.co.uk](https://www.football-data.co.uk), plus upcoming fixtures.
2. **Model** — Dixon-Coles (1997) bivariate Poisson with low-score dependence correction and exponential time decay (half-life ≈ 1 year), fit per league by weighted MLE.
3. **Predict** — daily GitHub Actions run issues forecasts for fixtures in the next 7 days and commits them to [`predictions/`](predictions/). First issuance stands; nothing is overwritten.
4. **Evaluate** — once results arrive, forecasts are scored with Brier score and log loss against de-vigged closing-line probabilities (Pinnacle closing preferred).

## Usage

```bash
pip install -e .
closingline ingest      # download data
closingline predict     # issue forecasts for the next 7 days
closingline evaluate    # score issued forecasts vs results and the market
```

## Roadmap

- [x] v1: Dixon-Coles baseline, daily automated forecasts, Brier/log-loss evaluation
- [ ] Public dashboard (Next.js) with live calibration plots and league tables
- [ ] Handle promoted teams properly (train on second-division data)
- [ ] Historical backtest suite across past seasons
- [ ] Model zoo: Elo-Poisson, gradient-boosted models, ensemble
- [ ] Expected-value analysis vs opening vs closing lines (CLV study)

## Disclaimer

This is a statistical modeling and market-efficiency research project. It is **not betting advice**, and no wagering is involved or endorsed.

---

Built by [Jonah Alsfasser](https://jonahx-dev.vercel.app) · Data © [football-data.co.uk](https://www.football-data.co.uk)
