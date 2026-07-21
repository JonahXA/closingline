import raw from "../public/data.json";
import Calibration from "../components/Calibration";
import BrierBars from "../components/BrierBars";
import Monthly from "../components/Monthly";

type Summary = {
  model?: string;
  league: string;
  matches: number;
  model_brier: number;
  market_brier: number;
  model_logloss: number;
  market_logloss: number;
};

const MODEL_NAMES: Record<string, string> = {
  "dixon-coles": "Dixon-Coles",
  "elo-poisson": "Elo-Poisson",
  gbm: "Gradient boosting (form + xG)",
  "xg-dixon-coles": "Dixon-Coles on goals/xG blend",
  ensemble: "Ensemble (weighted log-pool)",
};

type Upcoming = {
  Div: string;
  Date: string;
  HomeTeam: string;
  AwayTeam: string;
  p_home: number;
  p_draw: number;
  p_away: number;
  generated_at: string;
};

const data = raw as unknown as {
  generated_at: string;
  leagues: Record<string, string>;
  backtest: {
    primary_model?: string;
    summary: Summary[];
    models?: Summary[];
    calibration: { bin_mid: number; predicted: number; observed: number; n: number }[];
    monthly: { month: string; model_brier: number; market_brier: number; n: number }[];
    start: string;
    end: string;
  } | null;
  live: { upcoming: Upcoming[]; scored: unknown[]; summary?: Summary[] };
  clv?: {
    matches: number;
    model_brier: number;
    opening_brier: number;
    closing_brier: number;
    movement_corr: number;
    movement_sign_agreement: number;
  };
  significance?: {
    comparison: string;
    metric: string;
    matches: number;
    mean_diff: number;
    ci_low: number;
    ci_high: number;
    p_bootstrap: number;
    dm_stat: number;
    p_dm: number;
  }[];
};

const pct = (v: number) => `${Math.round(v * 100)}%`;

function SummaryTable({ rows, names }: { rows: Summary[]; names: Record<string, string> }) {
  return (
    <table>
      <thead>
        <tr>
          <th>League</th>
          <th>Matches</th>
          <th>Model Brier</th>
          <th>Market Brier</th>
          <th>Model log loss</th>
          <th>Market log loss</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.league}>
            <td>{r.league === "ALL" ? <strong>All leagues</strong> : names[r.league] ?? r.league}</td>
            <td>{r.matches}</td>
            <td className={r.model_brier <= r.market_brier ? "win" : ""}>{r.model_brier.toFixed(4)}</td>
            <td className={r.market_brier < r.model_brier ? "win" : ""}>{r.market_brier.toFixed(4)}</td>
            <td className={r.model_logloss <= r.market_logloss ? "win" : ""}>{r.model_logloss.toFixed(4)}</td>
            <td className={r.market_logloss < r.model_logloss ? "win" : ""}>{r.market_logloss.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Home() {
  const bt = data.backtest;
  const all = bt?.summary.find((s) => s.league === "ALL");
  const gap = all ? ((all.model_brier - all.market_brier) / all.market_brier) * 100 : null;

  return (
    <main>
      <div className="hero">
        <h1>ClosingLine</h1>
        <p className="tagline">
          Live probabilistic forecasts for the Big-5 European football leagues —{" "}
          <strong>frozen in git before kickoff</strong>, never revised, and publicly benchmarked
          against the sharpest number in sport: the sportsbook closing line.
        </p>
      </div>

      {all && gap !== null && (
        <div className="stat-row">
          <div className="stat-tile">
            <div className="value">{all.matches.toLocaleString()}</div>
            <div className="label">matches in walk-forward backtest</div>
          </div>
          <div className="stat-tile">
            <div className="value">{all.model_brier.toFixed(3)}</div>
            <div className="label">model Brier score (lower is better)</div>
          </div>
          <div className="stat-tile">
            <div className="value">{all.market_brier.toFixed(3)}</div>
            <div className="label">closing-line Brier score</div>
          </div>
          <div className="stat-tile">
            <div className="value">+{gap.toFixed(1)}%</div>
            <div className="label">gap to the market — the number this project exists to shrink</div>
          </div>
        </div>
      )}

      <section className="card">
        <h2>Live forecasts</h2>
        <p className="sub">
          Issued daily by GitHub Actions and committed before kickoff — the git history is the
          tamper-evident timestamp.
        </p>
        {data.live.upcoming.length === 0 ? (
          <div className="empty">
            No fixtures in the next 7 days — the 2026–27 season kicks off in August. Forecasts will
            appear here automatically from matchweek 1.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>League</th>
                <th>Match</th>
                <th>Home</th>
                <th>Draw</th>
                <th>Away</th>
              </tr>
            </thead>
            <tbody>
              {data.live.upcoming.map((m, i) => (
                <tr key={i}>
                  <td>{m.Date}</td>
                  <td>{data.leagues[m.Div]}</td>
                  <td>
                    {m.HomeTeam} — {m.AwayTeam}
                  </td>
                  <td>{pct(m.p_home)}</td>
                  <td>{pct(m.p_draw)}</td>
                  <td>{pct(m.p_away)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {bt && (
        <>
          <section className="card">
            <h2>Calibration</h2>
            <p className="sub">
              Walk-forward backtest, {bt.start} to {bt.end}. Every forecast probability vs how often
              that outcome actually happened. A perfectly calibrated model sits on the dashed
              diagonal.
            </p>
            <Calibration bins={bt.calibration} />
          </section>

          <section className="card">
            <h2>Model vs the closing line, by league</h2>
            <p className="sub">
              Brier score per league (lower is better). The market wins everywhere — for now. The
              size of the gap is the research result.
            </p>
            <BrierBars rows={bt.summary} names={data.leagues} />
            <div style={{ marginTop: 18 }}>
              <SummaryTable rows={bt.summary} names={data.leagues} />
            </div>
          </section>

          {bt.models && bt.models.length > 1 && (
            <section className="card">
              <h2>Model zoo</h2>
              <p className="sub">
                All leagues pooled. The starred model drives the headline stats and charts above;
                every model&apos;s forecasts are published daily, so each builds its own public
                track record. The ensemble&apos;s weights are fit purely on out-of-sample
                history — currently dominated by Dixon-Coles-on-xG (the best single model,
                tuned by walk-forward sweep) with gradient boosting for the rest, and goals-only
                Dixon-Coles and Elo priced out of the pool.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Brier</th>
                    <th>Log loss</th>
                    <th>Brier gap to market</th>
                  </tr>
                </thead>
                <tbody>
                  {[...bt.models]
                    .sort((a, b) => a.model_brier - b.model_brier)
                    .map((m) => (
                      <tr key={m.model}>
                        <td>
                          {MODEL_NAMES[m.model ?? ""] ?? m.model}
                          {m.model === bt.primary_model ? " ★" : ""}
                        </td>
                        <td>{m.model_brier.toFixed(4)}</td>
                        <td>{m.model_logloss.toFixed(4)}</td>
                        <td>
                          +{(((m.model_brier - m.market_brier) / m.market_brier) * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  <tr>
                    <td>
                      <strong>Closing line (de-vigged)</strong>
                    </td>
                    <td>{bt.models[0].market_brier.toFixed(4)}</td>
                    <td>{bt.models[0].market_logloss.toFixed(4)}</td>
                    <td>—</td>
                  </tr>
                </tbody>
              </table>
            </section>
          )}

          <section className="card">
            <h2>Accuracy over time</h2>
            <p className="sub">
              Monthly Brier score, model vs de-vigged closing line. Spikes are usually
              season-opening months, when team strength is most uncertain.
            </p>
            <Monthly rows={bt.monthly} />
          </section>
        </>
      )}

      {data.clv && (
        <section className="card">
          <h2>Where the market&apos;s edge lives</h2>
          <p className="sub">
            Opening line (days before kickoff) vs closing line vs model, same{" "}
            {data.clv.matches.toLocaleString()} matches.
          </p>
          <table>
            <thead>
              <tr>
                <th>Forecast</th>
                <th>Brier</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Closing line</td>
                <td>{data.clv.closing_brier.toFixed(4)}</td>
              </tr>
              <tr>
                <td>Opening line</td>
                <td>{data.clv.opening_brier.toFixed(4)}</td>
              </tr>
              <tr>
                <td>ClosingLine model</td>
                <td>{data.clv.model_brier.toFixed(4)}</td>
              </tr>
            </tbody>
          </table>
          <p className="sub" style={{ marginTop: 14, marginBottom: 0 }}>
            Three implications. The market barely improves from open to close, so its edge is not
            last-minute team news — whatever it knows, it knows days early. Our model loses to the
            opening line almost as badly as to the close, so the missing information is already
            public well before kickoff. And the model does not predict line movement (correlation{" "}
            {data.clv.movement_corr.toFixed(2)}, sign agreement{" "}
            {Math.round(data.clv.movement_sign_agreement * 100)}% — a coin flip): there is no
            exploitable signal in disagreeing with the opening line.
          </p>
        </section>
      )}

      {data.significance && data.significance.length > 0 && (
        <section className="card">
          <h2>Is the gap real? Significance tests</h2>
          <p className="sub">
            Both claims tested on per-match Brier differentials over{" "}
            {data.significance[0].matches.toLocaleString()} matches — paired bootstrap (10,000
            resamples) for a distribution-free 95% CI, and a Diebold-Mariano test with a
            Newey-West (HAC) variance so same-round score correlation doesn&apos;t understate the
            error. A differential whose CI excludes zero is significant.
          </p>
          <div className="chart-scroll">
            <table>
              <thead>
                <tr>
                  <th>Comparison (Brier)</th>
                  <th>Mean diff</th>
                  <th>95% CI</th>
                  <th>p (bootstrap)</th>
                  <th>p (DM)</th>
                </tr>
              </thead>
              <tbody>
                {data.significance
                  .filter((s) => s.metric === "brier")
                  .map((s) => (
                    <tr key={s.comparison}>
                      <td>{s.comparison}</td>
                      <td>{s.mean_diff > 0 ? "+" : ""}{s.mean_diff.toFixed(4)}</td>
                      <td>
                        [{s.ci_low.toFixed(4)}, {s.ci_high.toFixed(4)}]
                      </td>
                      <td>{s.p_bootstrap < 0.001 ? "<0.001" : s.p_bootstrap.toFixed(3)}</td>
                      <td>{s.p_dm < 0.001 ? "<0.001" : s.p_dm.toFixed(3)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ marginTop: 14, marginBottom: 0 }}>
            The market&apos;s edge over the model is real, not noise (p &lt; 0.001) — so the
            residual gap is a genuine research target, not sampling luck. And fitting the classical
            model on xG instead of goals is a statistically significant improvement (CI excludes
            zero on both metrics), confirming the one feature change that moved the needle was a
            real gain rather than a lucky draw.
          </p>
        </section>
      )}

      <section className="card">
        <h2>Method</h2>
        <p className="sub" style={{ marginBottom: 0 }}>
          Four structurally different models — Dixon-Coles (1997) bivariate Poisson with
          low-score dependence correction and exponential time decay, the same structure fit on a
          50/50 goals/xG blend (its decay rate tuned by walk-forward sweep), an Elo-Poisson model
          (goal-margin-weighted Elo ratings mapped to expected goals by Poisson regression), and
          a gradient-boosted classifier over causal form and Understat xG features — combined in
          a log-linear pool whose weights are fit only on out-of-sample predictions. All fit per
          league on
          top-flight and second-division results, so promoted teams carry real parameters.
          Backtests are strictly walk-forward: each forecast uses only information available
          before its refit date. Market probabilities are de-vigged closing odds, Pinnacle
          preferred. This is a market-efficiency research project, not betting advice.
        </p>
      </section>

      <footer>
        Updated {new Date(data.generated_at).toUTCString()} · Built by{" "}
        <a href="https://jonahx-dev.vercel.app">Jonah Alsfasser</a> · Code &amp; forecast history on{" "}
        <a href="https://github.com/JonahXA/closingline">GitHub</a> · Data ©{" "}
        <a href="https://www.football-data.co.uk">football-data.co.uk</a>
      </footer>
    </main>
  );
}
