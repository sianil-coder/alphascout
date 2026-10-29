"""AlphaScout ADK Multi-Agent Package.

Exports `root_agent` as required by the Google Agent Development Kit (`adk web`
and `adk run`).
"""

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

__all__ = [
    "adversarial_debate_squad",
    "bull_thesis_agent",
    "cio_thesis_synthesizer_agent",
    "equity_research_pipeline",
    "insider_sentiment_agent",
    "parallel_market_research_squad",
    "quant_metrics_agent",
    "root_agent",
    "run_alphascout_research_workflow",
    "sec_filing_rag_agent",
    "short_seller_bear_agent",
]
