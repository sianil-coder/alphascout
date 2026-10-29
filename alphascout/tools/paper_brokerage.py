"""Alpaca Paper Brokerage Execution Tool with Mandatory Human-in-the-Loop Gate.

Provides safe virtual brokerage order execution:
- Human-in-the-Loop Gate: High-stakes paper trade execution
  (`execute_paper_brokerage_order`) enforces an explicit code stop requiring
  human confirmation (`human_approved=True`) before submitting an order.
- Cloud Secret Management: Retrieves Alpaca Paper API credentials via
  `SecretManagerVault` (`google.cloud.secretmanager_v1`) with automatic
  fallback to the local simulated paper brokerage ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
import urllib.request
import uuid

from alphascout.config import AlphaScoutConfig, SecretManagerVault
from alphascout.memory.session_store import (
    async_memory_consolidator,
    load_seed_data,
)
from alphascout.observability.telemetry import (
    structured_logger,
    traced_tool_execution,
)
from alphascout.schemas import (
    GuidedErrorResponse,
    PaperOrderExecutionInput,
    PaperOrderReceipt,
    ValidationError,
)
from alphascout.tools.market_data import fetch_live_stock_fundamentals

# In-memory mutable paper portfolio state initialized from seed data
_RUNTIME_PAPER_LEDGER: dict[str, Any] | None = None


def get_paper_portfolio_state() -> dict[str, Any]:
    """Returns the current virtual paper brokerage portfolio ledger."""
    global _RUNTIME_PAPER_LEDGER
    if _RUNTIME_PAPER_LEDGER is None:
        seed = load_seed_data()
        _RUNTIME_PAPER_LEDGER = dict(seed.get("paper_portfolio_ledger", {}))
    return _RUNTIME_PAPER_LEDGER


def reset_paper_portfolio_state() -> None:
    """Resets the in-memory paper brokerage ledger back to initial seed state."""
    global _RUNTIME_PAPER_LEDGER
    _RUNTIME_PAPER_LEDGER = None


def _submit_to_alpaca_paper_api(
    config: AlphaScoutConfig,
    vault: SecretManagerVault,
    order_req: PaperOrderExecutionInput,
) -> dict[str, Any] | None:
    """Submits a live paper order to Alpaca Paper Brokerage if secrets are configured."""
    api_key = vault.get_secret(
        config.alpaca_api_key_secret_id,
        env_fallback_var="ALPACA_PAPER_API_KEY",
    )
    secret_key = vault.get_secret(
        config.alpaca_secret_key_secret_id,
        env_fallback_var="ALPACA_PAPER_SECRET_KEY",
    )
    if not api_key or not secret_key:
        return None

    url = f"{config.alpaca_paper_base_url.rstrip('/')}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "symbol": order_req.ticker.upper(),
        "qty": str(order_req.quantity),
        "side": order_req.side,
        "type": order_req.order_type,
        "time_in_force": "day",
    }
    if order_req.order_type == "limit" and order_req.limit_price:
        payload["limit_price"] = str(order_req.limit_price)

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "order_id": data.get(
                        "id", f"ALP-{uuid.uuid4().hex[:8].upper()}"
                    ),
                    "filled_avg_price": float(
                        data.get("filled_avg_price") or 0.0
                    ),
                    "venue": "Alpaca Paper Brokerage API (Live)",
                }
    except Exception as exc:
        structured_logger.warning(
            f"Alpaca API network request fallback triggered: {exc}",
            extra={"event_type": "BROKERAGE_FALLBACK"},
        )
    return None


def execute_paper_brokerage_order(
    ticker: str,
    side: str,
    quantity: int,
    rationale_summary: str,
    order_type: str = "market",
    limit_price: float | None = None,
    human_approved: bool = False,
) -> dict[str, Any]:
    """Executes or stages a virtual stock order on the Paper Brokerage account.

    CRITICAL HUMAN-IN-THE-LOOP (HITL) SAFETY RULE:
    This tool modifies portfolio holdings and therefore enforces an explicit
    Human-in-the-Loop confirmation gate. If `human_approved` is `False` (the
    default), this tool HALTS execution before any trade is placed and returns
    `status="pending_human_confirmation"` with a confirmation prompt to display
    to the human user. Only set `human_approved=True` after the user has
    explicitly approved the exact ticker, quantity, and estimated dollar amount.

    Args:
        ticker: Stock ticker symbol to trade (e.g., 'NVDA').
        side: Order direction, either 'buy' or 'sell'.
        quantity: Whole number of shares to trade (1 to 10,000).
        rationale_summary: Concise summary of the research thesis (10 to 500 chars).
        order_type: Execution type, 'market' or 'limit' (default: 'market').
        limit_price: Optional limit price in USD when `order_type='limit'`.
        human_approved: Explicit Human-in-the-Loop confirmation flag. Defaults to
            `False`. Must only be `True` after explicit human sign-off.

    Returns:
        A dictionary conforming to `PaperOrderReceipt` showing either
        `status="pending_human_confirmation"` (when awaiting human approval) or
        `status="filled"` with the execution order ID, fill price, and updated
        paper cash balance.
    """
    raw_args = {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "rationale_summary": rationale_summary,
        "order_type": order_type,
        "limit_price": limit_price,
        "human_approved": human_approved,
    }
    with traced_tool_execution(
        "execute_paper_brokerage_order",
        raw_args,
        agent_name="alphascout_coordinator",
    ) as trace_state:
        try:
            validated = PaperOrderExecutionInput(
                ticker=ticker.strip(),
                side=side.strip().lower(),  # type: ignore
                quantity=quantity,
                order_type=order_type.strip().lower(),  # type: ignore
                limit_price=limit_price,
                rationale_summary=rationale_summary.strip(),
                human_approved=human_approved,
            )
        except ValidationError as exc:
            err = GuidedErrorResponse(
                error_code="INVALID_PAPER_ORDER_SCHEMA",
                message=f"Paper brokerage order validation failed: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure ticker is valid, side is 'buy' or 'sell', quantity is an "
                    "integer >= 1, and rationale_summary is at least 10 characters."
                ),
            ).model_dump()
            trace_state["status"] = "error"
            trace_state["outcome"] = err
            return err

        norm_ticker = validated.ticker.upper()
        fundamentals = fetch_live_stock_fundamentals(norm_ticker)
        if fundamentals.get("status") == "error":
            trace_state["status"] = "error"
            trace_state["outcome"] = fundamentals
            return fundamentals

        execution_price = (
            float(validated.limit_price)
            if validated.order_type == "limit" and validated.limit_price
            else float(fundamentals["current_price"])
        )
        notional_usd = round(validated.quantity * execution_price, 2)
        ledger = get_paper_portfolio_state()
        cash_balance = float(ledger.get("cash_balance_usd", 100000.0))

        # ====================================================================
        # Explicit Human-in-the-Loop (HITL) Execution Gate
        # ====================================================================
        if not validated.human_approved:
            hitl_receipt = PaperOrderReceipt(
                status="pending_human_confirmation",
                order_id=None,
                broker_execution_venue="Alpaca Paper Brokerage (HITL Gate Active)",
                ticker=norm_ticker,
                side=validated.side,
                quantity=validated.quantity,
                filled_avg_price=execution_price,
                notional_value_usd=notional_usd,
                remaining_cash_usd=cash_balance,
                human_confirmation_prompt=(
                    f"[HUMAN-IN-THE-LOOP CONFIRMATION REQUIRED] Ready to submit "
                    f"PAPER {validated.side.upper()} order for {validated.quantity} "
                    f"shares of {norm_ticker} at ~${execution_price:,.2f}/share "
                    f"(Estimated Notional: ${notional_usd:,.2f}). Please ask the user "
                    f"to explicitly confirm before re-calling with human_approved=True."
                ),
                recovery_instructions=(
                    "Present the confirmation prompt to the human user and wait for "
                    "their explicit approval before calling execute_paper_brokerage_order "
                    "with human_approved=True."
                ),
            ).model_dump()
            trace_state["status"] = "pending_human_confirmation"
            trace_state["outcome"] = hitl_receipt
            return hitl_receipt

        if validated.side == "buy" and notional_usd > cash_balance:
            max_affordable = int(cash_balance // execution_price)
            err = GuidedErrorResponse(
                error_code="INSUFFICIENT_PAPER_BUYING_POWER",
                message=(
                    f"Order notional (${notional_usd:,.2f}) exceeds available paper "
                    f"cash balance (${cash_balance:,.2f})."
                ),
                recovery_instructions=(
                    f"Reduce order quantity to at most {max_affordable} shares or call "
                    "calculate_risk_adjusted_order_size to compute a cash-compliant size."
                ),
            ).model_dump()
            trace_state["status"] = "error"
            trace_state["outcome"] = err
            return err

        # Attempt live Alpaca Paper API via Secret Manager credentials, fallback to local ledger
        config = AlphaScoutConfig()
        vault = SecretManagerVault(project_id=config.project_id)
        alpaca_res = _submit_to_alpaca_paper_api(config, vault, validated)

        order_id = (
            alpaca_res["order_id"]
            if alpaca_res
            else f"ALP-PAPER-{uuid.uuid4().hex[:8].upper()}"
        )
        venue = (
            alpaca_res["venue"]
            if alpaca_res
            else "Alpaca Paper Brokerage (Local Sandbox Ledger)"
        )
        if alpaca_res and alpaca_res.get("filled_avg_price", 0.0) > 0:
            execution_price = alpaca_res["filled_avg_price"]
            notional_usd = round(validated.quantity * execution_price, 2)

        # Update paper ledger state
        if validated.side == "buy":
            cash_balance = round(cash_balance - notional_usd, 2)
        else:
            cash_balance = round(cash_balance + notional_usd, 2)

        ledger["cash_balance_usd"] = cash_balance
        positions = ledger.setdefault("positions", {})
        pos = positions.setdefault(
            norm_ticker, {"shares": 0, "avg_cost_usd": execution_price}
        )
        if validated.side == "buy":
            pos["shares"] = int(pos.get("shares", 0)) + validated.quantity
        else:
            pos["shares"] = max(0, int(pos.get("shares", 0)) - validated.quantity)

        order_record = {
            "order_id": order_id,
            "ticker": norm_ticker,
            "side": validated.side,
            "quantity": validated.quantity,
            "filled_avg_price": execution_price,
            "notional_value_usd": notional_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        ledger.setdefault("order_history", []).append(order_record)

        # Trigger non-blocking async memory consolidation for the executed trade
        async_memory_consolidator.schedule_memory_consolidation(
            session_id="paper-brokerage-session",
            ticker=norm_ticker,
            raw_interaction_notes=(
                f"Executed paper {validated.side.upper()} of {validated.quantity} "
                f"shares of {norm_ticker} @ ${execution_price:.2f} (Order ID: {order_id}). "
                f"Thesis: {validated.rationale_summary}"
            ),
            metadata=order_record,
        )

        receipt = PaperOrderReceipt(
            status="filled",
            order_id=order_id,
            broker_execution_venue=venue,
            ticker=norm_ticker,
            side=validated.side,
            quantity=validated.quantity,
            filled_avg_price=execution_price,
            notional_value_usd=notional_usd,
            remaining_cash_usd=cash_balance,
            human_confirmation_prompt=None,
        ).model_dump()
        trace_state["outcome"] = receipt
        return receipt
