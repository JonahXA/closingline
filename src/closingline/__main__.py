"""CLI: closingline {ingest,predict,evaluate}."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="closingline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Download historical results and upcoming fixtures")
    p = sub.add_parser("predict", help="Issue forecasts for upcoming fixtures")
    p.add_argument("--horizon", type=int, default=7, help="Days ahead to predict")
    sub.add_parser("evaluate", help="Score issued forecasts vs results and the market")
    b = sub.add_parser("backtest", help="Walk-forward backtest over past seasons")
    b.add_argument("--seasons", type=int, default=3)
    b.add_argument("--refit-days", type=int, default=28)
    sub.add_parser("export", help="Export dashboard data JSON from reports and predictions")
    sub.add_parser("clv", help="Closing-line-value study from the backtest report")
    sub.add_parser("bias", help="Scan the backtest for market soft spots by bucket")
    sw = sub.add_parser("sweep", help="Walk-forward hyperparameter sweep")
    sw.add_argument("--model", choices=["xgdc", "gbm", "elo", "shrinkage"], default="xgdc")
    sub.add_parser("significance", help="Paired bootstrap + Diebold-Mariano tests on score gaps")
    sub.add_parser("squad", help="Squad-strength ratings from prior-season player xG")
    sub.add_parser(
        "oracle",
        help="Upper-bound study: value of lineup info (uses post-match data, NOT live-usable)",
    )
    p2 = sub.add_parser("paper", help="Log/settle hypothetical value bets (no real wagering)")
    p2.add_argument("--settle", action="store_true", help="Score settled bets instead of logging")

    args = parser.parse_args()
    if args.command == "ingest":
        from . import data, xg

        data.download_history()
        data.download_fixtures()
        xg.download_xg()
        xg.download_players()
        print("Ingest complete.")
    elif args.command == "predict":
        from . import predict

        out = predict.run(horizon_days=args.horizon)
        if out.empty:
            print("No new fixtures to predict in the horizon window.")
        else:
            print(f"Issued {len(out)} forecasts:")
            print(out.to_string(index=False))
    elif args.command == "evaluate":
        from . import evaluate

        evaluate.run()
    elif args.command == "backtest":
        from . import backtest

        backtest.run(seasons=args.seasons, refit_days=args.refit_days)
    elif args.command == "export":
        from . import export

        export.run()
    elif args.command == "clv":
        from . import clv

        clv.run()
    elif args.command == "bias":
        from . import bias

        bias.run()
    elif args.command == "sweep":
        from . import sweep

        {
            "xgdc": sweep.run,
            "gbm": sweep.run_gbm,
            "elo": sweep.run_elo,
            "shrinkage": sweep.run_shrinkage,
        }[args.model]()
    elif args.command == "significance":
        from . import significance

        significance.run()
    elif args.command == "oracle":
        from . import oracle

        oracle.run()
    elif args.command == "squad":
        from pathlib import Path

        from .squad import team_features

        feats = team_features()
        Path("reports").mkdir(exist_ok=True)
        feats.to_csv("reports/squad.csv", index=False)
        latest = feats[feats["season"] == feats["season"].max()]
        print(latest.sort_values("squad_strength", ascending=False).head(15).to_string(index=False))
        print(f"\nWrote reports/squad.csv ({len(feats)} team-seasons).")
    elif args.command == "paper":
        from . import paper

        if args.settle:
            paper.settle()
        else:
            out = paper.log_bets()
            print(f"Logged {len(out)} hypothetical bets." if not out.empty else "No value edges found.")


if __name__ == "__main__":
    main()
