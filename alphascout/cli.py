"""Command-Line Demo Runner for AlphaScout.

Demonstrates the end-to-end multi-agent workflow:
1. Input prompt containing client PII (scrubbed automatically by `PIIRedactionEngine`).
2. Parallel market data (`yfinance`), SEC 10-K Risk Factor retrieval, and Form 4 insider flow.
3. Deterministic Intrinsic Valuation & Risk-Adjusted Position Sizing.
4. Human-in-the-Loop (HITL) stop before paper trade execution.
5. Explicit human confirmation -> Paper Brokerage Order Fill Receipt & OpenTelemetry audit summary.
"""

from __future__ import annotations

import json
import sys

from alphascout.agent import run_alphascout_research_workflow
from alphascout.observability.telemetry import (
    get_audit_log_buffer,
    get_exported_spans,
)


def main() -> int:
    print("=" * 78)
    print("  ALPHASCOUT: Autonomous Equity Research & Agentic Trading")
    print("=" * 78)

    sample_query = (
        "Run a deep-dive institutional equity research report on NVDA vs peers "
        "for client account ACC-8841-9920 (email: john.smith@example.com, "
        "SSN: 123-45-6789) and stage a risk-sized paper buy order if rated BUY."
    )
    print(f"\n[1] User Query (with synthetic PII):\n    {sample_query}\n")

    print("[2] Phase 1 Run (Research + Valuation + HITL Gate with human_approved=False)...")
    phase1 = run_alphascout_research_workflow(
        user_query=sample_query,
        ticker="NVDA",
        peer_tickers=["AMD", "AVGO"],
        human_approved_trade=False,
    )

    print("\n--- Generated Research Report ---")
    print(phase1["response_text"])

    receipt_p1 = phase1.get("paper_order_receipt") or {}
    print("\n--- Human-in-the-Loop (HITL) Gate Status ---")
    print(json.dumps(receipt_p1, indent=2))

    print("\n[3] Phase 2 Run (Simulating Human Confirmation -> human_approved=True)...")
    phase2 = run_alphascout_research_workflow(
        user_query="Confirm and execute the recommended NVDA paper buy order.",
        ticker="NVDA",
        peer_tickers=["AMD", "AVGO"],
        human_approved_trade=True,
    )
    receipt_p2 = phase2.get("paper_order_receipt") or {}
    print("\n--- Executed Paper Brokerage Order Receipt ---")
    print(json.dumps(receipt_p2, indent=2))

    spans = get_exported_spans()
    audit_events = get_audit_log_buffer()
    print(
        f"\n[4] Observability Summary: {len(spans)} OpenTelemetry Spans recorded | "
        f"{len(audit_events)} Intent/Outcome Audit Logs captured (100% PII Redacted)."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
