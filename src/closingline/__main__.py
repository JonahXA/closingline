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

    args = parser.parse_args()
    if args.command == "ingest":
        from . import data

        data.download_history()
        data.download_fixtures()
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


if __name__ == "__main__":
    main()
