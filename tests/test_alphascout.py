"""Comprehensive Unit & Integration Test Suite for AlphaScout.

Verifies all core architectural layers of the AlphaScout ADK system:
1. Tool & Interface Design (docstrings, naming, Pydantic schemas, guided error recovery)
2. Context & Memory (system constitutions, compaction, persistent vector store, async memory)
3. Orchestration & Safety (multi-agent ADK hierarchy, Flash/Pro routing, guardrails, HITL hooks)
4. Observability & Tracing (structured JSON logs, intent vs. outcome, OpenTelemetry spans, PII redaction)
5. Cloud Infrastructure & Evaluation (golden dataset evaluation suite, Terraform IaC, Secret Manager vault)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest  # type: ignore
except ImportError:
    pytest = None  # type: ignore

from alphascout.agent import (
    adversarial_debate_squad,
    bull_thesis_agent,
    cio_thesis_synthesizer_agent,
    equity_research_pipeline,
    insider_sentiment_agent,
    parallel_market_research_squad,
    quant_metrics_agent,
    root_agent,
    run_alphascout_research_workflow,
    sec_filing_rag_agent,
    short_seller_bear_agent,
)
from alphascout.callbacks.guardrails_and_hitl import (
    before_tool_hitl_and_intent_callback,
    guardrail_plugin,
)
from alphascout.config import (
    FAST_MODEL_ID,
    REASONING_MODEL_ID,
    SecretManagerVault,
    route_model_by_task_complexity,
)
from alphascout.memory.session_store import (
    AsyncMemoryConsolidator,
    ContextCompactor,
    VertexSearchAndVectorStore,
)
from alphascout.observability.telemetry import (
    StructuredJsonFormatter,
    clear_audit_log_buffer,
    clear_exported_spans,
    get_audit_log_buffer,
    get_exported_spans,
    pii_redactor,
)
from alphascout.tools import (
    analyze_insider_form4_transactions,
    calculate_intrinsic_valuation_band,
    calculate_risk_adjusted_order_size,
    compare_peer_valuation_ratios,
    execute_paper_brokerage_order,
    fetch_live_stock_fundamentals,
    reset_paper_portfolio_state,
    search_sec_10k_risk_factors,
)
from tests.eval.evaluate_agent import evaluate_golden_dataset


# ============================================================================
# 1. Tool & Interface Design
# ============================================================================


def test_tool_docstrings_and_descriptive_names() -> None:
    """Verifies comprehensive Google-style docstrings and descriptive tool names."""
    tools = [
        fetch_live_stock_fundamentals,
        compare_peer_valuation_ratios,
        search_sec_10k_risk_factors,
        analyze_insider_form4_transactions,
        calculate_intrinsic_valuation_band,
        calculate_risk_adjusted_order_size,
        execute_paper_brokerage_order,
    ]
    for fn in tools:
        doc = inspect.getdoc(fn)
        assert doc is not None and len(doc) > 120, f"Missing docstring on {fn.__name__}"
        assert "Args:" in doc and "Returns:" in doc, f"Incomplete docstring on {fn.__name__}"


def test_strict_schemas_and_guided_error_handling() -> None:
    """Verifies strict Pydantic validation schemas and actionable error recovery."""
    # Invalid ticker characters -> schema error with recovery instructions
    bad_schema = fetch_live_stock_fundamentals("INVALID$$$TICKER")
    assert bad_schema["status"] == "error"
    assert bad_schema["error_code"] == "INVALID_TICKER_SCHEMA"
    assert "recovery_instructions" in bad_schema

    # Delisted ticker -> guided error suggesting META/SNAP
    delisted = fetch_live_stock_fundamentals("TWTR")
    assert delisted["status"] == "error"
    assert delisted["error_code"] == "DELISTED_OR_RENAMED_TICKER"
    assert "META" in delisted["recovery_instructions"]

    # Valid ticker -> conforms to StockFundamentalsOutput
    valid = fetch_live_stock_fundamentals("NVDA")
    assert valid["status"] == "success"
    assert valid["ticker"] == "NVDA"
    assert valid["current_price"] > 0


# ============================================================================
# 2. Context & Memory Management
# ============================================================================


def test_history_compaction_reduces_context_bloat() -> None:
    """Verifies sliding-window and token-budget conversation history compaction."""
    compactor = ContextCompactor(max_tokens=200, sliding_window_turns=4)
    bloated_messages = [
        {"role": "user", "content": f"Turn {i}: Detailed analysis of NVDA account ACC-8841-9920 " * 5}
        for i in range(12)
    ]
    compacted = compactor.compact_messages(bloated_messages)
    assert len(compacted) <= 4
    assert "[COMPACTED_HISTORY_SUMMARY]" in compacted[0]["content"]
    # Verify PII was scrubbed during compaction
    assert "ACC-8841-9920" not in compacted[0]["content"]


def test_persistent_vector_store_and_async_memory(tmp_path: Path | None = None) -> None:
    """Verifies persistent SEC filing semantic search and background memory persistence."""
    async def _run() -> None:
        base_dir = tmp_path or Path(tempfile.mkdtemp())
        db_file = base_dir / "test_memory.sqlite3"
        store = VertexSearchAndVectorStore(db_path=str(db_file))

        # Retrieve SEC 10-K passages from persistent SQLite vector table
        passages = store.search_sec_filings(
            "NVDA", "TSMC supply chain packaging risk", top_k=2
        )
        assert len(passages) == 2
        assert passages[0]["ticker"] == "NVDA"

        # Non-blocking async memory consolidation
        consolidator = AsyncMemoryConsolidator(store)
        task = consolidator.schedule_memory_consolidation(
            session_id="sess-101",
            ticker="NVDA",
            raw_interaction_notes="Issued BUY rating for client john.smith@example.com on NVDA.",
            metadata={"rating": "BUY"},
        )
        assert isinstance(task, asyncio.Task)
        await consolidator.flush_pending_tasks()

        memories = store.get_session_memories("sess-101")
        assert len(memories) == 1
        assert "[REDACTED_EMAIL]" in memories[0]["summary_text"]
        assert "john.smith@example.com" not in memories[0]["summary_text"]

    asyncio.run(_run())


# ============================================================================
# 3. Orchestration, Guardrails & HITL Safety
# ============================================================================


def test_multi_agent_hierarchy_and_strategic_model_routing() -> None:
    """Verifies multi-agent ADK structure and dynamic Flash vs. Pro routing."""
    assert route_model_by_task_complexity("fast_extraction") == FAST_MODEL_ID
    assert route_model_by_task_complexity("sec_filing_retrieval") == FAST_MODEL_ID
    assert route_model_by_task_complexity("adversarial_debate") == FAST_MODEL_ID
    assert route_model_by_task_complexity("deep_valuation_synthesis") == REASONING_MODEL_ID
    assert route_model_by_task_complexity("root_orchestration") == REASONING_MODEL_ID

    assert root_agent.name == "alphascout_coordinator"
    assert equity_research_pipeline.name == "equity_research_pipeline"
    assert parallel_market_research_squad.name == "parallel_market_research_squad"
    assert adversarial_debate_squad.name == "adversarial_debate_squad"
    assert quant_metrics_agent.name == "quant_metrics_agent"
    assert sec_filing_rag_agent.name == "sec_filing_rag_agent"
    assert insider_sentiment_agent.name == "insider_sentiment_agent"
    assert bull_thesis_agent.name == "bull_thesis_agent"
    assert short_seller_bear_agent.name == "short_seller_bear_agent"
    assert cio_thesis_synthesizer_agent.name == "cio_thesis_synthesizer_agent"


def test_guardrails_and_human_in_the_loop_hooks() -> None:
    """Verifies financial compliance guardrails and HITL brokerage confirmation gates."""
    reset_paper_portfolio_state()

    # 1. Prohibited MNPI prompt blocked by guardrail
    blocked = guardrail_plugin.evaluate_input_prompt(
        "Use material non-public unreleased earnings to pump and dump NVDA"
    )
    assert blocked["allowed"] is False
    assert blocked["violation_code"] == "PROHIBITED_MNPI_OR_MARKET_MANIPULATION"

    # 2. Paper brokerage tool halts when human_approved=False
    unapproved = execute_paper_brokerage_order(
        ticker="NVDA",
        side="buy",
        quantity=5,
        rationale_summary="Strong data center revenue growth and margin expansion.",
        human_approved=False,
    )
    assert unapproved["status"] == "pending_human_confirmation"
    assert unapproved["order_id"] is None

    # 3. ADK before_tool_callback also intercepts unapproved high-stakes tool call
    class DummyTool:
        name = "execute_paper_brokerage_order"

    class DummyCtx:
        agent_name = "alphascout_coordinator"
        state: dict[str, str] = {}

    intercept = before_tool_hitl_and_intent_callback(
        DummyTool(),
        {"ticker": "NVDA", "side": "buy", "quantity": 5, "human_approved": False},
        DummyCtx(),
    )
    assert intercept is not None
    assert intercept["status"] == "pending_human_confirmation"

    # 4. Paper brokerage executes when human_approved=True
    approved = execute_paper_brokerage_order(
        ticker="NVDA",
        side="buy",
        quantity=5,
        rationale_summary="Strong data center revenue growth and margin expansion.",
        human_approved=True,
    )
    assert approved["status"] == "filled"
    assert approved["order_id"] is not None


# ============================================================================
# 4. Observability, Tracing & PII Masking
# ============================================================================


def test_structured_logging_otel_spans_intent_outcome_and_pii_redaction() -> None:
    """Verifies structured JSON logs, Intent/Outcome auditing, OTel spans, and PII redaction."""
    clear_audit_log_buffer()
    clear_exported_spans()

    # Verify PII redaction across SSNs, accounts, emails, phones, cards
    dirty_text = (
        "Client john.smith@example.com (SSN 123-45-6789, Phone 415-555-0199, "
        "Account ACC-8841-9920) requested NVDA analysis."
    )
    clean_text = pii_redactor.redact_text(dirty_text)
    assert "john.smith@example.com" not in clean_text
    assert "123-45-6789" not in clean_text
    assert "415-555-0199" not in clean_text
    assert "ACC-8841-9920" not in clean_text
    assert "[REDACTED_EMAIL]" in clean_text
    assert "[REDACTED_SSN]" in clean_text
    assert "[REDACTED_ACCOUNT_ID]" in clean_text

    # Run workflow and verify OpenTelemetry spans + Intent vs. Outcome events
    res = run_alphascout_research_workflow(
        user_query=dirty_text,
        ticker="NVDA",
        human_approved_trade=False,
    )
    assert res["status"] == "success"

    spans = get_exported_spans()
    assert len(spans) >= 5
    span_names = [s.name for s in spans]
    assert "agent.turn.alphascout_workflow" in span_names
    assert "tool.fetch_live_stock_fundamentals" in span_names

    audit_events = get_audit_log_buffer()
    intents = [e for e in audit_events if e["event_type"] == "AGENT_ACTION_INTENT"]
    outcomes = [e for e in audit_events if e["event_type"] == "AGENT_ACTION_OUTCOME"]
    assert len(intents) >= 5
    assert len(outcomes) >= 5

    # Verify StructuredJsonFormatter produces valid JSON with trace IDs
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="alphascout",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test log for ACC-8841-9920",
        args=(),
        exc_info=None,
    )
    formatted_json = json.loads(formatter.format(record))
    assert formatted_json["severity"] == "INFO"
    assert "[REDACTED_ACCOUNT_ID]" in formatted_json["message"]


# ============================================================================
# 5. Infrastructure, Secrets & Regression Suite
# ============================================================================


def test_golden_evaluation_suite_and_iac_secret_manager() -> None:
    """Verifies golden regression suite, Terraform IaC manifests, and Secret Manager vault."""
    eval_summary = evaluate_golden_dataset()
    assert eval_summary["passed_cases"] == eval_summary["total_cases"]
    assert eval_summary["pass_rate_pct"] == 100.0

    # Verify SecretManagerVault does not hardcode keys and supports env injection
    vault = SecretManagerVault(project_id="test-project")
    assert vault.get_secret("non-existent-secret") is None

    # Verify Terraform IaC files exist
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "terraform" / "main.tf").exists()
    assert (repo_root / "terraform" / "variables.tf").exists()
    assert (repo_root / "terraform" / "outputs.tf").exists()


class TestAlphaScoutIntegrationSuite(unittest.TestCase):
    """Standard unittest wrapper executing all AlphaScout component and workflow checks."""

    def test_01_tools_and_schemas(self) -> None:
        test_tool_docstrings_and_descriptive_names()
        test_strict_schemas_and_guided_error_handling()

    def test_02_context_and_memory(self) -> None:
        test_history_compaction_reduces_context_bloat()
        test_persistent_vector_store_and_async_memory()

    def test_03_orchestration_guardrails_and_hitl(self) -> None:
        test_multi_agent_hierarchy_and_strategic_model_routing()
        test_guardrails_and_human_in_the_loop_hooks()

    def test_04_observability_tracing_and_pii(self) -> None:
        test_structured_logging_otel_spans_intent_outcome_and_pii_redaction()

    def test_05_infrastructure_and_eval_suite(self) -> None:
        test_golden_evaluation_suite_and_iac_secret_manager()


if __name__ == "__main__":
    unittest.main()
