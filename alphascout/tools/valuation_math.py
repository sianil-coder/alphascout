"""Deterministic Valuation Band & Risk-Adjusted Position Sizing Tools.

This module implements deterministic financial mathematics so the LLM never
performs mental arithmetic for target prices or portfolio share allocation:
1. `calculate_intrinsic_valuation_band`: Computes Bear, Base, and Bull target
   prices using forward EPS growth and peer-relative P/E multiples.
2. `calculate_risk_adjusted_order_size`: Sizes paper trade share counts against
   the client's virtual portfolio equity and stock volatility (Beta).
"""

from __future__ import annotations

import math
from typing import Any
from alphascout.memory.session_store import load_seed_data
from alphascout.observability.telemetry import traced_tool_execution
from alphascout.schemas import (
    GuidedErrorResponse,
    OrderSizeInput,
    OrderSizeOutput,
    ValidationError,
    ValuationBandInput,
    ValuationBandOutput,
)


def calculate_intrinsic_valuation_band(
    ticker: str,
    current_price: float,
    trailing_eps: float,
    expected_eps_growth_pct: float,
    peer_median_pe: float,
) -> dict[str, Any]:
    """Calculates deterministic Bear, Base, and Bull intrinsic price targets.

    Combines 2-year forward compounded EPS projection with growth-adjusted peer
    multiples to compute mathematically exact target prices and implied upside
    percentages.

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA').
        current_price: Current market share price in USD (must be > 0).
        trailing_eps: Trailing 12-month diluted EPS in USD.
        expected_eps_growth_pct: Projected annual EPS growth rate percentage
            between -50.0 and 200.0 (e.g., 35.0 for 35% annual growth).
        peer_median_pe: Sector peer median forward P/E multiple (between 1.0 and 150.0).

    Returns:
        A dictionary conforming to `ValuationBandOutput` with Bear/Base/Bull
        price targets and a valuation signal (`UNDERVALUED`, `FAIRLY_VALUED`, or
        `OVERVALUED`), or a `GuidedErrorResponse` on invalid inputs.
    """
    raw_args = {
        "ticker": ticker,
        "current_price": current_price,
        "trailing_eps": trailing_eps,
        "expected_eps_growth_pct": expected_eps_growth_pct,
        "peer_median_pe": peer_median_pe,
    }
    with traced_tool_execution(
        "calculate_intrinsic_valuation_band",
        raw_args,
        agent_name="quant_metrics_agent",
    ) as trace_state:
        try:
            validated = ValuationBandInput(
                ticker=ticker.strip(),
                current_price=current_price,
                trailing_eps=trailing_eps,
                expected_eps_growth_pct=expected_eps_growth_pct,
                peer_median_pe=peer_median_pe,
            )
        except ValidationError as exc:
            err = GuidedErrorResponse(
                error_code="INVALID_VALUATION_PARAMETERS",
                message=f"Valuation calculation input error: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure current_price > 0, expected_eps_growth_pct is between "
                    "-50 and 200, and peer_median_pe is between 1 and 150. Call "
                    "fetch_live_stock_fundamentals first if you need current_price or EPS."
                ),
            ).model_dump()
            trace_state["status"] = "error"
            trace_state["outcome"] = err
            return err

        if validated.trailing_eps <= 0:
            err = GuidedErrorResponse(
                error_code="NON_POSITIVE_EPS",
                message=f"Trailing EPS ({validated.trailing_eps}) must be positive for P/E multiple valuation.",
                recovery_instructions=(
                    "Pass a positive normalized forward EPS estimate or use "
                    "fetch_live_stock_fundamentals to obtain diluted EPS."
                ),
            ).model_dump()
            trace_state["status"] = "error"
            trace_state["outcome"] = err
            return err

        growth_rate = validated.expected_eps_growth_pct / 100.0
        # 12-month forward NTM EPS projection
        ntm_eps_growth = max(-0.12, min(0.32, growth_rate * 0.75))
        forward_ntm_eps = round(validated.trailing_eps * (1.0 + ntm_eps_growth), 2)

        # Institutional 12-Month Target Exit Multiple:
        # Quality compounders (P/E <= 45x) retain ~98% of their multiple as NTM EPS grows,
        # while hyper-growth high-P/E names (>45x) experience modest multiple compression.
        current_pe = validated.current_price / validated.trailing_eps
        multiple_retention = max(
            0.94, min(1.02, 0.985 - max(0.0, current_pe - 42.0) * 0.0035)
        )
        peg_support_pe = validated.peer_median_pe * max(
            1.02, min(1.75, 1.08 + (growth_rate - 0.15) * 1.75)
        )
        blended_target_pe = round(
            (current_pe * multiple_retention * 0.82) + (peg_support_pe * 0.18), 2
        )

        raw_base_target = forward_ntm_eps * blended_target_pe
        # Institutional 12-month price target guardrail (-20% to +32% from spot price)
        base_target = round(
            max(
                validated.current_price * 0.80,
                min(validated.current_price * 1.32, raw_base_target),
            ),
            2,
        )
        bear_target = round(validated.current_price * 0.85, 2)
        bull_target = round(base_target * 1.15, 2)

        upside_pct = round(
            ((base_target - validated.current_price) / validated.current_price)
            * 100.0,
            2,
        )

        if upside_pct >= 10.0:
            signal = "UNDERVALUED"
        elif upside_pct <= -10.0:
            signal = "OVERVALUED"
        else:
            signal = "FAIRLY_VALUED"

        output = ValuationBandOutput(
            status="success",
            ticker=validated.ticker.upper(),
            current_price=validated.current_price,
            bear_case_target=bear_target,
            base_case_target=base_target,
            bull_case_target=bull_target,
            implied_base_upside_pct=upside_pct,
            valuation_signal=signal,
            methodology_notes=(
                f"12M Forward NTM EPS (${forward_ntm_eps:.2f}, +{ntm_eps_growth*100:.1f}% NTM growth) "
                f"at blended {blended_target_pe:.1f}x target multiple (vs peer median {validated.peer_median_pe:.1f}x)."
            ),
        ).model_dump()
        trace_state["outcome"] = output
        return output


def calculate_risk_adjusted_order_size(
    ticker: str,
    current_price: float,
    conviction_score: int,
    stock_beta: float = 1.2,
    max_portfolio_risk_pct: float = 2.0,
) -> dict[str, Any]:
    """Calculates the risk-adjusted number of shares to buy for the paper portfolio.

    Reads the client's virtual paper brokerage cash balance and portfolio equity,
    scales the allocation by conviction score (`0-100`) and inverse volatility
    (`1 / sqrt(beta)`), and returns the exact whole number of shares to stage.

    Args:
        ticker: Stock ticker symbol to size (e.g., 'NVDA').
        current_price: Current share price in USD (must be > 0).
        conviction_score: CIO investment conviction score from 0 to 100.
        stock_beta: Stock volatility beta relative to the S&P 500 (default: 1.2).
        max_portfolio_risk_pct: Maximum percentage of total portfolio equity to
            allocate to a single trade (between 0.1 and 10.0, default: 2.0).

    Returns:
        A dictionary conforming to `OrderSizeOutput` specifying `recommended_shares`,
        `estimated_notional_usd`, and `portfolio_allocation_pct`, or a
        `GuidedErrorResponse` on invalid inputs.
    """
    raw_args = {
        "ticker": ticker,
        "current_price": current_price,
        "conviction_score": conviction_score,
        "stock_beta": stock_beta,
        "max_portfolio_risk_pct": max_portfolio_risk_pct,
    }
    with traced_tool_execution(
        "calculate_risk_adjusted_order_size",
        raw_args,
        agent_name="cio_thesis_synthesizer_agent",
    ) as trace_state:
        try:
            validated = OrderSizeInput(
                ticker=ticker.strip(),
                current_price=current_price,
                conviction_score=conviction_score,
                stock_beta=stock_beta,
                max_portfolio_risk_pct=max_portfolio_risk_pct,
            )
        except ValidationError as exc:
            err = GuidedErrorResponse(
                error_code="INVALID_ORDER_SIZING_INPUT",
                message=f"Order sizing validation error: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure current_price > 0, conviction_score is an integer 0-100, "
                    "stock_beta is 0.1-5.0, and max_portfolio_risk_pct is 0.1-10.0."
                ),
            ).model_dump()
            trace_state["status"] = "error"
            trace_state["outcome"] = err
            return err

        seed = load_seed_data()
        ledger = seed.get("paper_portfolio_ledger", {})
        cash_usd = float(ledger.get("cash_balance_usd", 100000.0))
        equity_usd = float(ledger.get("total_portfolio_equity_usd", 125000.0))
        positions = ledger.get("positions", {})
        norm_ticker = validated.ticker.upper()
        existing_shares = int(
            positions.get(norm_ticker, {}).get("shares", 0)
        )

        # Conviction below 60 does not warrant a BUY allocation
        if validated.conviction_score < 60:
            output = OrderSizeOutput(
                status="success",
                ticker=norm_ticker,
                recommended_shares=0,
                estimated_notional_usd=0.0,
                portfolio_allocation_pct=0.0,
                available_cash_usd=cash_usd,
                existing_position_shares=existing_shares,
                sizing_rationale=(
                    f"Conviction score ({validated.conviction_score}/100) is below the "
                    "60/100 institutional threshold for capital deployment."
                ),
            ).model_dump()
            trace_state["outcome"] = output
            return output

        conviction_scalar = validated.conviction_score / 100.0
        volatility_scalar = 1.0 / math.sqrt(max(0.5, validated.stock_beta))
        target_pct = (
            validated.max_portfolio_risk_pct
            * conviction_scalar
            * volatility_scalar
        )
        target_notional = min(cash_usd * 0.95, equity_usd * (target_pct / 100.0))
        shares = max(1, int(target_notional // validated.current_price))
        actual_notional = round(shares * validated.current_price, 2)
        actual_pct = round((actual_notional / equity_usd) * 100.0, 2)

        output = OrderSizeOutput(
            status="success",
            ticker=norm_ticker,
            recommended_shares=shares,
            estimated_notional_usd=actual_notional,
            portfolio_allocation_pct=actual_pct,
            available_cash_usd=cash_usd,
            existing_position_shares=existing_shares,
            sizing_rationale=(
                f"Sized {shares} shares (${actual_notional:,.2f}, {actual_pct}% of "
                f"${equity_usd:,.0f} equity) using conviction={validated.conviction_score}/100 "
                f"and beta={validated.stock_beta:.2f} volatility adjustment."
            ),
        ).model_dump()
        trace_state["outcome"] = output
        return output
