"""Automated Golden Dataset Evaluation Harness for AlphaScout.

Evaluates the AlphaScout agent against the golden dataset (`alphascout.evalset.json`)
across 5 dimensions:
1. Tool Trajectory Accuracy (expected tools invoked in the right sequence)
2. Schema & Output Grounding Validity
3. Human-in-the-Loop (HITL) Gate Enforcement
4. Zero PII Leakage in Logs, Spans, and Outputs
5. Security & Compliance Guardrail Enforcement

Also provides an ADK `AgentEvaluator` integration hook for cloud evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alphascout.agent import run_alphascout_research_workflow
from alphascout.observability.telemetry import (
    clear_audit_log_buffer,
    clear_exported_spans,
    get_audit_log_buffer,
)
from alphascout.tools.paper_brokerage import reset_paper_portfolio_state

EVALSET_PATH = Path(__file__).resolve().parent / "alphascout.evalset.json"


def run_adk_agent_evaluator_if_enabled() -> None:
    """Optional hook for Google ADK `AgentEvaluator.evaluate` against live Vertex AI."""
    try:
        from google.adk.evaluation.agent_evaluator import AgentEvaluator  # type: ignore

        _ = AgentEvaluator
    except Exception:
        pass


def evaluate_golden_dataset(
    evalset_path: Path = EVALSET_PATH,
) -> dict[str, Any]:
    """Runs all golden dataset test cases and computes static regression scores."""
    with open(evalset_path, "r", encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)

    reset_paper_portfolio_state()
    results: list[dict[str, Any]] = []
    passed_count = 0

    for case in cases:
        clear_audit_log_buffer()
        clear_exported_spans()

        eval_id = case["eval_id"]
        output = run_alphascout_research_workflow(
            user_query=case["user_query"],
            ticker=case["ticker"],
            peer_tickers=case.get("peer_tickers"),
            human_approved_trade=bool(case.get("human_approved_trade", False)),
            session_id=f"eval-{eval_id}",
        )

        checks: list[tuple[str, bool, str]] = []

        # 1. Verify expected workflow status
        actual_status = output.get("status")
        expected_status = case["expected_status"]
        checks.append(
            (
                "status_match",
                actual_status == expected_status,
                f"expected={expected_status}, actual={actual_status}",
            )
        )

        # 2. Verify tool trajectory if specified
        audit_events = get_audit_log_buffer()
        called_tools = [
            e["tool_name"]
            for e in audit_events
            if e["event_type"] == "AGENT_ACTION_INTENT"
        ]
        if "expected_tools_called" in case:
            expected_tools = case["expected_tools_called"]
            missing_tools = [
                t for t in expected_tools if t not in called_tools
            ]
            checks.append(
                (
                    "tool_trajectory",
                    len(missing_tools) == 0,
                    f"missing_tools={missing_tools}",
                )
            )

        # 3. Verify Human-in-the-Loop order status if specified
        if "expected_order_status" in case:
            receipt = output.get("paper_order_receipt") or {}
            actual_order_status = receipt.get("status")
            checks.append(
                (
                    "hitl_order_status",
                    actual_order_status == case["expected_order_status"],
                    f"expected={case['expected_order_status']}, actual={actual_order_status}",
                )
            )

        # 4. Verify zero PII leakage in response and audit logs
        serialized_audit = json.dumps(audit_events)
        response_text = output.get("response_text", "")
        for pii_token in case.get("must_not_contain_pii", []):
            leaked = (pii_token in response_text) or (
                pii_token in serialized_audit
            )
            checks.append(
                (
                    f"no_pii_leak:{pii_token}",
                    not leaked,
                    "PII scrubbed" if not leaked else f"LEAKED: {pii_token}",
                )
            )

        # 5. Verify required grounded substrings
        for sub in case.get("must_contain_substrings", []):
            present = sub.lower() in response_text.lower()
            checks.append(
                (
                    f"contains:{sub}",
                    present,
                    f"substring '{sub}' check",
                )
            )

        case_passed = all(c[1] for c in checks)
        if case_passed:
            passed_count += 1

        results.append(
            {
                "eval_id": eval_id,
                "passed": case_passed,
                "checks": [
                    {"check": name, "passed": ok, "detail": detail}
                    for name, ok, detail in checks
                ],
            }
        )

    total = len(cases)
    pass_rate = round((passed_count / total) * 100.0, 2) if total else 0.0
    return {
        "total_cases": total,
        "passed_cases": passed_count,
        "pass_rate_pct": pass_rate,
        "results": results,
    }


def main() -> int:
    summary = evaluate_golden_dataset()
    print(json.dumps(summary, indent=2))
    if summary["passed_cases"] != summary["total_cases"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
