# ClosingLine

**Live, pre-registered probabilistic football forecasts, publicly benchmarked against the sportsbook closing line.**

**Dashboard: [jonahxa.github.io/closingline](https://jonahxa.github.io/closingline/)**

Every day, an automated pipeline fits Dixon-Coles models on the Big-5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), issues win/draw/loss probabilities for upcoming fixtures, and commits them to this repository **before the matches are played**. Git history is the timestamp: forecasts are frozen at first issuance and never revised, so the track record is tamper-evident and fully reproducible.

The research question, extending my [ML market-efficiency study of the 2026 World Cup](https://jonahx-dev.vercel.app): *how close can open statistical models get to the calibration of the closing line — the sharpest publicly observable probability estimate in sport?*

## How it works

1. **Ingest** — historical results and closing odds from [football-data.co.uk](https://www.football-data.co.uk), plus upcoming fixtures.
2. **Model** — a zoo of three structurally different models: Dixon-Coles (1997) bivariate Poisson with low-score dependence correction and exponential time decay; Elo-Poisson (margin-weighted Elo ratings mapped to expected goals); and a gradient-boosted classifier over causal form features (rolling goals, points per game, rest days, Elo diff). They are pooled in a log-linear ensemble whose weights are fit only on out-of-sample predictions — earlier walk-forward windows in the backtest, the committed backtest report live — so the weighting never sees in-sample fits. Every model's forecasts are published so each builds its own track record.
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
| Dixon-Coles | 0.5874 | 0.9855 |
| Ensemble (weighted log-pool, primary) | 0.5877 | 0.9859 |
| Elo-Poisson | 0.5911 | 0.9914 |
| Gradient boosting (form features) | 0.5953 | 0.9983 |
| De-vigged closing line | 0.5734 | 0.9644 |

The market wins everywhere — for now. The size of the gap is the research result, and shrinking it is the roadmap.

### Findings so far

1. **A well-built classical model gets within ~2.5% of the market, and is well calibrated** — the market's edge is information, not math.
2. **Equal-weight ensembling dilutes the best model; OOS-fitted weights fix it** — they converge to ~100% Dixon-Coles on their own. The ensemble stays primary because it self-corrects if any component starts adding value.
3. **Neither generic form features nor shots-on-target form add information beyond goals** — the GBM earned ~0 pool weight both times. Recovering more of the market's edge needs data of a different kind (true xG, lineups).
4. **The market's edge is not late-breaking news** (CLV study, `closingline clv`): the opening line (Brier 0.5747) is nearly as sharp as the close (0.5734), our model loses to both by similar margins, and the model has zero ability to predict line movement (r = −0.02, sign agreement 48%). Whatever the market knows, it knows days before kickoff — and it is already public information our goals-only models fail to extract.

## Roadmap

- [x] v1: Dixon-Coles baseline, daily automated forecasts, Brier/log-loss evaluation
- [x] Public dashboard (Next.js + Recharts on GitHub Pages) with calibration plots and league tables
- [x] Handle promoted teams properly (train on second-division data)
- [x] Walk-forward backtest suite across past seasons
- [x] Model zoo: Elo-Poisson, gradient-boosted form model, weighted log-pool ensemble (weights fit on out-of-sample history only)
- [x] Unit tests (feature causality, weight fitting, de-vig) in CI
- [x] Shots-on-target features for the GBM (negative result — no added information)
- [x] CLV study: opening vs closing vs model, line-movement prediction
- [ ] True xG data (Understat/FBref) — the main remaining candidate for closing the gap
- [ ] Season-opening weakness: promotion-aware priors or early-season shrinkage toward market baseline
- [ ] Live-season report after matchweek 10: pre-registered forecasts vs the market

## Disclaimer

This is a statistical modeling and market-efficiency research project. It is **not betting advice**, and no wagering is involved or endorsed.

---

Built by [Jonah Alsfasser](https://jonahx-dev.vercel.app) · Data © [football-data.co.uk](https://www.football-data.co.uk)
