"""Callbacks package for AlphaScout."""

from alphascout.callbacks.guardrails_and_hitl import (
    FinancialComplianceGuardrailPlugin,
    HIGH_STAKES_TOOLS,
    MANDATORY_DISCLAIMER,
    after_model_self_eval_callback,
    after_tool_outcome_callback,
    before_model_guardrail_callback,
    before_tool_hitl_and_intent_callback,
    guardrail_plugin,
)

__all__ = [
    "FinancialComplianceGuardrailPlugin",
    "HIGH_STAKES_TOOLS",
    "MANDATORY_DISCLAIMER",
    "after_model_self_eval_callback",
    "after_tool_outcome_callback",
    "before_model_guardrail_callback",
    "before_tool_hitl_and_intent_callback",
    "guardrail_plugin",
]
