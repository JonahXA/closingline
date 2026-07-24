# ClosingLine

**Live, pre-registered probabilistic football forecasts, publicly benchmarked against the sportsbook closing line.**

**Dashboard: [jonahxa.github.io/closingline](https://jonahxa.github.io/closingline/)**

Every day, an automated pipeline fits Dixon-Coles models on the Big-5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), issues win/draw/loss probabilities for upcoming fixtures, and commits them to this repository **before the matches are played**. Git history is the timestamp: forecasts are frozen at first issuance and never revised, so the track record is tamper-evident and fully reproducible.

The research question, extending my [ML market-efficiency study of the 2026 World Cup](https://jonahx-dev.vercel.app): *how close can open statistical models get to the calibration of the closing line — the sharpest publicly observable probability estimate in sport?*

## How it works

1. **Ingest** — historical results and closing odds from [football-data.co.uk](https://www.football-data.co.uk), per-match xG from [Understat](https://understat.com), plus upcoming fixtures.
2. **Model** — a zoo of four structurally different models: Dixon-Coles (1997) bivariate Poisson with low-score dependence correction and exponential time decay; the same structure fit on a goals/xG blend (`xg-dixon-coles`, the dominant component); Elo-Poisson (margin-weighted Elo ratings mapped to expected goals); and a gradient-boosted classifier over causal form and xG features (rolling goals, points per game, shots on target, per-match xG, rest days, Elo diff). They are pooled in a log-linear ensemble whose weights are fit only on out-of-sample predictions — earlier walk-forward windows in the backtest, the committed backtest report live — so the weighting never sees in-sample fits. Hyperparameters (blend ratio, time-decay rate) are tuned by walk-forward sweep (`closingline sweep`). Every model's forecasts are published so each builds its own track record.
3. **Predict** — daily GitHub Actions run issues forecasts for fixtures in the next 7 days and commits them to [`predictions/`](predictions/). First issuance stands; nothing is overwritten.
4. **Evaluate** — once results arrive, forecasts are scored with Brier score and log loss against de-vigged closing-line probabilities (Pinnacle closing preferred).

## Usage

```bash
pip install -e .
closingline ingest      # download data
closingline predict     # issue forecasts for the next 7 days
closingline evaluate    # score issued forecasts vs results and the market
closingline backtest    # walk-forward backtest over past seasons
closingline sweep       # walk-forward hyperparameter sweep (blend ratio, decay)
closingline bias        # scan the backtest for market soft spots
closingline clv         # closing-line-value study (opening vs closing vs model)
closingline paper       # log/settle hypothetical value bets (research only, no wagering)
closingline export      # regenerate dashboard/public/data.json
```

The dashboard is a static Next.js app in [`dashboard/`](dashboard/), rebuilt and deployed to GitHub Pages on every push.

## Backtest results (3 seasons, 5,286 matches, walk-forward)

| | Brier | Log loss |
|---|---|---|
| **Ensemble (weighted log-pool, primary)** | **0.5854** | **0.9826** |
| Dixon-Coles on goals/xG blend (tuned) | 0.5850 | 0.9821 |
| Dixon-Coles | 0.5874 | 0.9855 |
| Elo-Poisson | 0.5911 | 0.9914 |
| Gradient boosting (form + xG features) | 0.5931 | 0.9955 |
| De-vigged closing line | 0.5734 | 0.9644 |

Pool weights (fit on out-of-sample history only): ~80–100% xG-blend Dixon-Coles, ~16–20% GBM, goals-only Dixon-Coles and Elo priced out to 0.

The market wins everywhere — for now. The size of the gap is the research result, and shrinking it is the roadmap.

### Findings so far

1. **A well-built classical model gets within ~2.5% of the market, and is well calibrated** — the market's edge is information, not math.
2. **Equal-weight ensembling dilutes the best model; OOS-fitted weights fix it** — they converge to ~100% Dixon-Coles on their own. The ensemble stays primary because it self-corrects if any component starts adding value.
3. **Neither generic form features nor shots-on-target form add information beyond goals** — the GBM earned ~0 pool weight both times. **True xG does**: with Understat per-match xG features the GBM earns 25–29% of the pool in every league, and the ensemble beats Dixon-Coles everywhere (0.5867 vs 0.5874) — the first model improvement over the classical baseline, cutting the gap to the market from +2.44% to +2.32%.
4. **The market's edge is not late-breaking news** (CLV study, `closingline clv`): the opening line (Brier 0.5747) is nearly as sharp as the close (0.5734), our model loses to both by similar margins, and the model has zero ability to predict line movement (r = −0.02, sign agreement 48%). Whatever the market knows, it knows days before kickoff — and it is already public information our goals-only models fail to extract.
5. **Fitting the classical model on chance quality beats fitting it on goals**: Dixon-Coles on a 50/50 goals/xG blend beats goals-only Dixon-Coles in every league, and the OOS pool weights immediately made it the dominant component. A walk-forward hyperparameter sweep (`closingline sweep`) confirmed the 50/50 blend was already optimal but found the classical time-decay default (xi=0.0019) too slow — xi=0.003 (recent matches weighted more heavily) improved it further to 0.5850, and the tuned model now takes 80–100% of the pool. Cumulative gap to the market: +2.44% → **+2.09%** across four model generations.
6. **Prior-season squad quality adds nothing either** (`closingline squad`): minutes-weighted, credibility-shrunk player xG ratings from completed prior seasons rank teams correctly (Liverpool/City/Arsenal top; promoted sides bottom with 36–46% squad continuity), but adding them to the GBM made it *worse* (0.5931 → 0.5945) and cost it pool weight (23% → 17%). Squad quality is largely redundant with what Elo and rolling xG already encode — a good squad produces good xG — so it added variance without information. The module is retained for lineup-aware work, where *per-match availability* rather than season aggregates is the real signal.
7. **Shrinkage doesn't help — the model already shrinks itself** (`closingline sweep --model shrinkage`). Empirical-Bayes shrinkage on the team ratings (an L2 penalty toward the league mean, the textbook fix for the season-opening weakness) made the model *monotonically worse* at every strength (0.5850 → 0.6034). Why: two mechanisms already do shrinkage's job — the tuned time-decay down-weights thin evidence, and fitting on xG rather than goals is itself a lower-variance signal. With no estimator variance left to trade away, the penalty only adds bias. The season-opening weakness is genuine uncertainty about new teams, not fixable variance.
8. **Even perfect lineup knowledge would not close the gap** (`closingline oracle`). Lineup-aware forecasting needs *confirmed* lineups published ~60 min before kickoff — a paid feed we don't have; Understat's per-match rosters are post-match records, and training on them would be leakage. So instead of faking it, we measured an **upper bound**: give the model an oracle it could never have in production (who actually took the field) and ask whether that explains its errors. On 380 EPL matches, oracle lineup strength correlates **+0.31 with actual goal difference** — the measure is genuinely informative, better lineups do win — but only **+0.02 with the model's residual error**. Elo and rolling xG already extract that information. This closes the lineup avenue on evidence rather than on cost: a subscription would not have bought an edge.
9. **The market's soft spots are not where intuition says** (`closingline bias`): the gap is *smallest* in Aug–Sep (+0.009) and grows to +0.014 by spring — both we and the market are worst at season start, and the market's relative edge accumulates with in-season information. The EPL line is the sharpest (+0.020), Ligue 1 the softest (+0.008). No bucket flips the sign, and an EV simulation against fair closing prices loses 10–15% per unit — larger model-market disagreement predicts model error, not market error.

## Roadmap

- [x] v1: Dixon-Coles baseline, daily automated forecasts, Brier/log-loss evaluation
- [x] Public dashboard (Next.js + Recharts on GitHub Pages) with calibration plots and league tables
- [x] Handle promoted teams properly (train on second-division data)
- [x] Walk-forward backtest suite across past seasons
- [x] Model zoo: Elo-Poisson, gradient-boosted form model, weighted log-pool ensemble (weights fit on out-of-sample history only)
- [x] Unit tests (feature causality, weight fitting, de-vig) in CI
- [x] Shots-on-target features for the GBM (negative result — no added information)
- [x] CLV study: opening vs closing vs model, line-movement prediction
- [x] True xG data (Understat) — GBM earns ~26% pool weight; ensemble beats Dixon-Coles in every league
- [x] xG-blend Dixon-Coles — best single model, dominant in the pool
- [x] Bias scan (`closingline bias`) and pre-registered paper trading (`closingline paper`, quarter-Kelly, no real wagering)
- [x] Walk-forward hyperparameter sweep (`closingline sweep`) — tuned blend ratio + time decay (gap +2.21% → +2.09%)
- [x] Second tuning pass — GBM (0.5931 → 0.5902) and Elo (0.5911 → 0.5906) tuned; shrinkage tested and rejected. Both component gains are individually significant, but the **ensemble is unchanged within noise** (Δ −0.0001, p = 0.43): the dominant xG-Dixon-Coles was already tuned, and improving minor pool members doesn't move the pool. Kept for the per-model track records; not claimed as an ensemble gain.
- [x] Per-player season xG/xA capture (Understat) — foundation for player-level ratings
- [x] Player-level xG squad ratings (`squad.py`) — negative result: redundant with Elo + xG, removed from the GBM
- [x] Lineup-aware forecasting (`closingline oracle`) — closed on evidence: an upper-bound study shows even perfect lineup knowledge is near-orthogonal to the model's errors
- [ ] **Live-season track record** — with the public-data feature avenues now exhausted, the pre-registered forecasts become the primary evidence stream. Report due after ~matchweek 10.

## Where this leaves the research question

Four model generations closed the gap from +2.44% to **+2.09%**, and every remaining candidate has now been tested and closed on evidence rather than left as an open promise:

| Tried | Verdict |
|---|---|
| Rolling form (goals, points, rest) | No information beyond goals |
| Shots on target | No information beyond goals |
| **True xG** | **Real gain** (p < 0.001) |
| **xG-blended Dixon-Coles + tuned decay** | **Real gain**, now dominant in the pool |
| Prior-season squad quality | Redundant with Elo + xG |
| Perfect (oracle) lineup knowledge | Near-orthogonal to model error |
| Empirical-Bayes shrinkage | Worse — model already self-shrinks via decay + xG |
| Elo / GBM hyperparameter tuning | Component gains significant; ensemble unchanged (p = 0.43) |

On the paper track record: it is a **measurement instrument, not a strategy**. Simulated against fair closing prices the same rule loses 10.8% per unit at a 3% edge filter and *more* as the filter tightens (−33.8% at 50%) — larger claimed edge means larger model error. It runs live to test one thing out-of-sample: whether the closing line moves toward our positions. That is the only result that would reopen the edge question, and the backtest says it does not.

The consistent pattern: **shot quality was the only genuinely new information**; everything describing *who is on the pitch* was already priced into the ratings by the results themselves. The residual gap to the closing line is not explained by any public-data feature we have been able to construct — which is itself the project's central empirical result, and is consistent with the market's edge coming from information that is not in box scores at all.

## Disclaimer

This is a statistical modeling and market-efficiency research project. It is **not betting advice**, and no wagering is involved or endorsed.

---

Built by [Jonah Alsfasser](https://jonahx-dev.vercel.app) · Data © [football-data.co.uk](https://www.football-data.co.uk)
