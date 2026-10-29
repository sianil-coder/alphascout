"""Strict Pydantic JSON Schemas for AlphaScout Tool Inputs, Outputs, and Reports.

All tool arguments, tool responses, and structured agent outputs are strictly
validated using Pydantic v2 `BaseModel` schemas with field constraints, regex
patterns, and descriptions to constrain LLM outputs and prevent hallucinations.
"""

from __future__ import annotations

import re
from typing import Any, Literal, get_args, get_origin

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError  # type: ignore
except ImportError:

    class ValidationError(ValueError):
        """Fallback validation error matching Pydantic v2 `.errors()` interface."""

        def __init__(self, msg: str) -> None:
            super().__init__(msg)
            self._msg = msg

        def errors(self) -> list[dict[str, str]]:
            return [{"msg": self._msg}]

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore
        return kwargs

    class _FieldSpec:
        def __init__(self, default: Any = ..., **kwargs: Any) -> None:
            self.default = default
            self.kwargs = kwargs

    def Field(default: Any = ..., **kwargs: Any) -> Any:  # type: ignore
        return _FieldSpec(default, **kwargs)

    class BaseModel:  # type: ignore
        """Fallback schema validator enforcing Field constraints & extra='forbid'."""

        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))

            if self.model_config.get("extra") == "forbid":
                unknown = set(data.keys()) - set(annotations.keys())
                if unknown:
                    raise ValidationError(f"Extra inputs are not permitted: {unknown}")

            for field_name in annotations:
                spec = getattr(self.__class__, field_name, None)
                if isinstance(spec, _FieldSpec):
                    if field_name in data:
                        val = data[field_name]
                    elif spec.default is not ...:
                        val = spec.default
                    else:
                        raise ValidationError(f"Field '{field_name}' is required")
                    self._validate_field(field_name, val, spec.kwargs)
                else:
                    if field_name in data:
                        val = data[field_name]
                    elif hasattr(self.__class__, field_name):
                        val = getattr(self.__class__, field_name)
                    else:
                        raise ValidationError(f"Field '{field_name}' is required")
                setattr(self, field_name, val)

        @staticmethod
        def _validate_field(name: str, val: Any, rules: dict[str, Any]) -> None:
            if val is None:
                return
            if "pattern" in rules and isinstance(val, str):
                if not re.match(rules["pattern"], val):
                    raise ValidationError(
                        f"String '{val}' does not match pattern {rules['pattern']}"
                    )
            if "min_length" in rules and hasattr(val, "__len__"):
                if len(val) < rules["min_length"]:
                    raise ValidationError(
                        f"Length of '{name}' must be >= {rules['min_length']}"
                    )
            if "max_length" in rules and hasattr(val, "__len__"):
                if len(val) > rules["max_length"]:
                    raise ValidationError(
                        f"Length of '{name}' must be <= {rules['max_length']}"
                    )
            if "gt" in rules and isinstance(val, (int, float)):
                if not (val > rules["gt"]):
                    raise ValidationError(f"'{name}' must be > {rules['gt']}")
            if "ge" in rules and isinstance(val, (int, float)):
                if not (val >= rules["ge"]):
                    raise ValidationError(f"'{name}' must be >= {rules['ge']}")
            if "le" in rules and isinstance(val, (int, float)):
                if not (val <= rules["le"]):
                    raise ValidationError(f"'{name}' must be <= {rules['le']}")

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)


# ============================================================================
# 1. Input Schemas (Constraining LLM Tool Arguments)
# ============================================================================


class StockFundamentalsInput(BaseModel):
    """Input schema for fetching live stock fundamentals."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description=(
            "Public equity ticker symbol (e.g., 'NVDA', 'AMD', 'MSFT', 'AAPL'). "
            "Must contain only letters, dots, or hyphens."
        ),
    )
    include_quarterly_financials: bool = Field(
        default=True,
        description="Whether to include trailing quarterly revenue and margin trends.",
    )


class PeerComparisonInput(BaseModel):
    """Input schema for comparing valuation ratios across peer companies."""

    model_config = ConfigDict(extra="forbid")

    primary_ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Primary target ticker symbol being analyzed (e.g., 'NVDA').",
    )
    peer_tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="List of 1 to 6 sector peer ticker symbols (e.g., ['AMD', 'INTC', 'AVGO']).",
    )


class SecRiskSearchInput(BaseModel):
    """Input schema for searching SEC 10-K/10-Q filings and risk factors."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Company ticker symbol for SEC EDGAR and vector search lookup.",
    )
    query_topic: str = Field(
        default="Item 1A Risk Factors supply chain competition regulatory",
        min_length=3,
        max_length=300,
        description="Specific topic or risk category to search inside 10-K/10-Q filings.",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of relevant SEC filing passages to return.",
    )


class InsiderTransactionsInput(BaseModel):
    """Input schema for analyzing SEC Form 4 executive insider buying and selling."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Company ticker symbol to inspect for SEC Form 4 insider activity.",
    )
    lookback_days: int = Field(
        default=180,
        ge=30,
        le=365,
        description="Lookback window in calendar days for executive Form 4 filings.",
    )


class ValuationBandInput(BaseModel):
    """Input schema for deterministic intrinsic value and target price calculation."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Stock ticker symbol.",
    )
    current_price: float = Field(
        ...,
        gt=0.0,
        description="Current share price in USD.",
    )
    trailing_eps: float = Field(
        ...,
        description="Trailing 12-month diluted Earnings Per Share (EPS) in USD.",
    )
    expected_eps_growth_pct: float = Field(
        ...,
        ge=-50.0,
        le=200.0,
        description="Projected annual EPS growth rate percentage (e.g., 25.0 for 25%).",
    )
    peer_median_pe: float = Field(
        ...,
        gt=1.0,
        le=150.0,
        description="Sector peer median forward P/E multiple.",
    )


class OrderSizeInput(BaseModel):
    """Input schema for risk-adjusted position sizing against portfolio constraints."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Ticker symbol to size.",
    )
    current_price: float = Field(
        ...,
        gt=0.0,
        description="Current market share price in USD.",
    )
    stock_beta: float = Field(
        default=1.2,
        gt=0.1,
        le=5.0,
        description="Stock beta measuring volatility relative to the S&P 500.",
    )
    conviction_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Investment conviction score from 0 to 100.",
    )
    max_portfolio_risk_pct: float = Field(
        default=2.0,
        gt=0.1,
        le=10.0,
        description="Maximum percentage of portfolio equity to allocate to this trade.",
    )


class PaperOrderExecutionInput(BaseModel):
    """Input schema for executing a paper trade order on Alpaca Paper Brokerage."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z.\-]+$",
        description="Ticker symbol to buy or sell in the paper brokerage account.",
    )
    side: Literal["buy", "sell"] = Field(
        ...,
        description="Order side: 'buy' or 'sell'.",
    )
    quantity: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Whole number of shares to trade.",
    )
    order_type: Literal["market", "limit"] = Field(
        default="market",
        description="Order execution type ('market' or 'limit').",
    )
    limit_price: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional limit price in USD when order_type is 'limit'.",
    )
    rationale_summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Concise summary of the research thesis justifying this paper order.",
    )
    human_approved: bool = Field(
        default=False,
        description=(
            "Explicit Human-in-the-Loop confirmation flag. Must be True only "
            "after the human user has explicitly approved the exact order details."
        ),
    )


# ============================================================================
# 2. Output Schemas (Constraining Tool Returns & Final Research Reports)
# ============================================================================


class GuidedErrorResponse(BaseModel):
    """Standardized guided error schema providing recovery instructions to the LLM."""

    status: Literal["error"] = "error"
    error_code: str = Field(..., description="Machine-readable error classification.")
    message: str = Field(..., description="Human-readable explanation of the failure.")
    recovery_instructions: str = Field(
        ...,
        description="Actionable instructions guiding the LLM on how to recover.",
    )


class StockFundamentalsOutput(BaseModel):
    """Validated output schema for live stock fundamentals."""

    status: Literal["success", "error"]
    ticker: str
    company_name: str = ""
    sector: str = ""
    current_price: float = 0.0
    market_cap_billions: float = 0.0
    trailing_pe: float | None = None
    forward_pe: float | None = None
    ev_to_ebitda: float | None = None
    revenue_growth_yoy_pct: float = 0.0
    gross_margin_pct: float = 0.0
    operating_margin_pct: float = 0.0
    free_cash_flow_billions: float = 0.0
    total_debt_to_equity: float | None = None
    beta: float = 1.0
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0
    analyst_target_mean: float | None = None
    data_source: str = "yfinance"
    recovery_instructions: str | None = None


class ValuationBandOutput(BaseModel):
    """Validated output schema for intrinsic valuation band calculations."""

    status: Literal["success", "error"]
    ticker: str
    current_price: float
    bear_case_target: float
    base_case_target: float
    bull_case_target: float
    implied_base_upside_pct: float
    valuation_signal: Literal["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"]
    methodology_notes: str
    recovery_instructions: str | None = None


class OrderSizeOutput(BaseModel):
    """Validated output schema for risk-adjusted order sizing."""

    status: Literal["success", "error"]
    ticker: str
    recommended_shares: int
    estimated_notional_usd: float
    portfolio_allocation_pct: float
    available_cash_usd: float
    existing_position_shares: int
    sizing_rationale: str
    recovery_instructions: str | None = None


class PaperOrderReceipt(BaseModel):
    """Validated output schema for paper brokerage order execution or HITL pause."""

    status: Literal["filled", "pending_human_confirmation", "error"]
    order_id: str | None = None
    broker_execution_venue: str = "Alpaca Paper Brokerage"
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    filled_avg_price: float | None = None
    notional_value_usd: float | None = None
    remaining_cash_usd: float | None = None
    human_confirmation_prompt: str | None = None
    recovery_instructions: str | None = None


class EquityResearchReportSchema(BaseModel):
    """Structured output schema for AlphaScout's final institutional research memo."""

    ticker: str = Field(..., description="Analyzed company ticker symbol.")
    company_name: str = Field(..., description="Full legal name of the company.")
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL"] = Field(
        ..., description="Official investment rating."
    )
    conviction_score: int = Field(
        ..., ge=0, le=100, description="Conviction score from 0 to 100."
    )
    current_price_usd: float = Field(..., gt=0.0)
    target_price_base_usd: float = Field(..., gt=0.0)
    implied_upside_pct: float = Field(...)
    bull_case_summary: list[str] = Field(
        ..., min_length=2, description="Key data-backed bull thesis pillars."
    )
    bear_case_and_10k_risks: list[str] = Field(
        ..., min_length=2, description="Key SEC 10-K risk factors and bear arguments."
    )
    insider_sentiment_verdict: str = Field(
        ..., description="Summary of SEC Form 4 executive buying/selling."
    )
    recommended_paper_trade_shares: int = Field(
        default=0, ge=0, description="Risk-sized share count if rating is BUY."
    )
    compliance_disclaimer: str = Field(
        ...,
        description="Mandatory SEC/FINRA non-advice regulatory disclaimer.",
    )
