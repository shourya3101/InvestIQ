import argparse
import json
from pathlib import Path


def cmd_analyze(args):
    from agents.coordinator_agent import run_full_analysis
    print(f"Analyzing {args.ticker.upper()}...")
    result = run_full_analysis(
        ticker=args.ticker,
        mode="live",
        days_back=args.days,
        run_debate_flag=True,
    )
    print(f"\n=== {result.ticker} Analysis ===")
    print(f"Action:     {result.memo.action.signal.upper()}")
    print(f"Confidence: {result.memo.action.confidence:.0%}")
    print(f"Risk:       {result.risk.risk_level.upper()} "
          f"({result.risk.risk_score:.0f}/100)")
    if result.debate:
        print(f"Debate:     {result.debate.final_bias}")
    print(f"\nThesis:")
    print(f"  {result.memo.thesis[:400]}")
    print(f"\nPipeline trace:")
    for step in result.pipeline_trace:
        print(f"  {step}")
    if result.memory:
        print(f"\nMemory: {result.memory.summary}")


def cmd_history(args):
    from agents.memory_agent import load_history
    ticker = args.ticker.upper()
    entries = load_history(ticker, n=10)
    if not entries:
        print(f"No history found for {ticker}")
        return
    print(f"\n=== {ticker} Analysis History ===")
    for e in entries:
        print(f"  {str(e.as_of)[:19]} | "
              f"risk={e.risk_score:.0f}/100 "
              f"({e.risk_level}) | "
              f"signal={e.action_signal.upper()} | "
              f"conf={e.confidence:.0%}")


def cmd_ingest(args):
    from core.document_loader import DocumentLoader
    from core.vector_store_manager import VectorStoreManager
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return
    loader = DocumentLoader()
    store = VectorStoreManager()
    if path.suffix == '.txt':
        docs = loader.load_txt(str(path), ticker=args.ticker)
    elif path.suffix == '.csv':
        docs = loader.load_csv(str(path), ticker=args.ticker)
        if not isinstance(docs, list):
            docs = [docs]
    else:
        print(f"Unsupported file type: {path.suffix}")
        return
    count = store.add_documents(docs)
    print(f"Ingested {count} document(s) "
          f"from {path.name} for "
          f"{(args.ticker or 'unknown').upper()}")


def main():
    parser = argparse.ArgumentParser(description="InvestIQ CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Run full analysis")
    p_analyze.add_argument("--ticker", required=True)
    p_analyze.add_argument("--days", type=int, default=365)

    p_history = sub.add_parser("history", help="View memo history")
    p_history.add_argument("--ticker", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("--file", required=True)
    p_ingest.add_argument("--ticker", default=None)

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "ingest":
        cmd_ingest(args)


if __name__ == "__main__":
    main()
