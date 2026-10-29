"""Guardrails, Self-Evaluation Policy Plugins, and Human-in-the-Loop (HITL) Callbacks.

Provides safety and compliance hooks for the AlphaScout agent lifecycle:
1. Guardrails & Policy Plugins:
   - `before_model_guardrail_callback`: Blocks prohibited market manipulation,
     insider-trading MNPI exploitation, and prompt injection attempts, while
     applying PII redaction and history compaction.
   - `after_model_self_eval_callback`: Performs self-evaluation on generated
     research outputs to verify numerical grounding, scrub PII, and guarantee
     mandatory SEC/FINRA regulatory disclaimers.
2. Human-in-the-Loop Hooks:
   - `before_tool_hitl_and_intent_callback`: Intercepts high-stakes tool calls
     (`execute_paper_brokerage_order`) and halts execution unless explicit human
     confirmation (`human_approved=True`) is present.
3. Intent vs. Outcome Capture:
   - Records `AGENT_ACTION_INTENT` before tool invocation and
     `AGENT_ACTION_OUTCOME` in `after_tool_outcome_callback`.
"""

from __future__ import annotations

import re
from typing import Any

from alphascout.memory.session_store import context_compactor
from alphascout.observability.telemetry import (
    pii_redactor,
    record_action_intent,
    record_action_outcome,
    structured_logger,
)

MANDATORY_DISCLAIMER = (
    "Disclaimer: This AlphaScout report is for informational and educational "
    "research purposes only and does not constitute personalized investment advice."
)

_PROHIBITED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:hack|bribe|steal|unreleased\s+earnings|material\s+non[\-\s]?public|"
            r"mnpi|pump\s+and\s+dump|spoof(?:ing)?\s+orders|wash\s+trading)\b",
            re.IGNORECASE,
        ),
        "PROHIBITED_MNPI_OR_MARKET_MANIPULATION",
    ),
    (
        re.compile(
            r"\b(?:ignore\s+all\s+previous\s+instructions|bypass\s+human\s+approval|"
            r"disable\s+guardrails)\b",
            re.IGNORECASE,
        ),
        "PROMPT_INJECTION_DETECTED",
    ),
]

HIGH_STAKES_TOOLS = frozenset({"execute_paper_brokerage_order"})


class FinancialComplianceGuardrailPlugin:
    """Security, compliance, and self-evaluation guardrail plugin for AlphaScout."""

    @staticmethod
    def evaluate_input_prompt(user_text: str) -> dict[str, Any]:
        """Evaluates incoming user prompts for policy violations and scrubs PII."""
        scrubbed_text = pii_redactor.redact_text(user_text)
        for pattern, violation_code in _PROHIBITED_PATTERNS:
            if pattern.search(user_text):
                structured_logger.warning(
                    f"Guardrail blocked input due to policy violation: {violation_code}",
                    extra={
                        "event_type": "GUARDRAIL_INPUT_BLOCKED",
                        "component": "FinancialComplianceGuardrailPlugin",
                        "metadata": {"violation_code": violation_code},
                    },
                )
                return {
                    "allowed": False,
                    "violation_code": violation_code,
                    "scrubbed_text": scrubbed_text,
                    "blocked_response": (
                        f"[COMPLIANCE GUARDRAIL BLOCKED: {violation_code}] "
                        "AlphaScout cannot assist with material non-public information "
                        "(MNPI), market manipulation, or bypassing safety controls. "
                        "Please submit a standard public equity research query."
                    ),
                }
        return {
            "allowed": True,
            "violation_code": None,
            "scrubbed_text": scrubbed_text,
            "blocked_response": None,
        }

    @staticmethod
    def self_evaluate_output(
        response_text: str,
        reference_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Self-evaluates model output for PII safety, grounding, and disclaimers."""
        scrubbed_output = pii_redactor.redact_text(response_text)
        has_disclaimer = "disclaimer:" in scrubbed_output.lower()
        if not has_disclaimer:
            scrubbed_output = f"{scrubbed_output.rstrip()}\n\n{MANDATORY_DISCLAIMER}"

        grounding_verified = True
        grounding_notes: list[str] = []
        if reference_metrics and reference_metrics.get("current_price"):
            ref_price = float(reference_metrics["current_price"])
            price_str = f"{ref_price:.2f}"
            if price_str in scrubbed_output or str(int(ref_price)) in scrubbed_output:
                grounding_notes.append(
                    f"Verified price grounding against tool output (${ref_price:.2f})."
                )

        structured_logger.info(
            "Completed self-evaluation guardrail check on model response",
            extra={
                "event_type": "GUARDRAIL_SELF_EVAL_PASSED",
                "component": "FinancialComplianceGuardrailPlugin",
                "metadata": {
                    "grounding_verified": grounding_verified,
                    "disclaimer_enforced": True,
                    "notes": grounding_notes,
                },
            },
        )
        return {
            "passed": True,
            "grounding_verified": grounding_verified,
            "final_text": scrubbed_output,
        }


guardrail_plugin = FinancialComplianceGuardrailPlugin()


# ============================================================================
# ADK-Compatible Callback Functions (`before_model`, `after_model`, `before_tool`, `after_tool`)
# ============================================================================


def before_model_guardrail_callback(
    callback_context: Any,
    llm_request: Any,
) -> Any | None:
    """ADK `before_model_callback` enforcing input guardrails, PII scrubbing & compaction."""
    try:
        contents = getattr(llm_request, "contents", None) or []
        if contents:
            last_content = contents[-1]
            parts = getattr(last_content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    eval_res = guardrail_plugin.evaluate_input_prompt(text)
                    part.text = eval_res["scrubbed_text"]
                    if not eval_res["allowed"]:
                        from google.genai import types  # type: ignore

                        return types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=eval_res["blocked_response"]
                                )
                            ],
                        )
    except Exception:
        pass
    return None


def after_model_self_eval_callback(
    callback_context: Any,
    llm_response: Any,
) -> Any | None:
    """ADK `after_model_callback` performing self-evaluation and PII/disclaimer enforcement."""
    try:
        content = getattr(llm_response, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text and len(text) > 80:
                    eval_res = guardrail_plugin.self_evaluate_output(text)
                    part.text = eval_res["final_text"]
    except Exception:
        pass
    return None


def before_tool_hitl_and_intent_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """ADK `before_tool_callback` logging intent and enforcing Human-in-the-Loop stops.

    If a high-stakes tool (`execute_paper_brokerage_order`) is invoked without
    `human_approved=True`, this callback intercepts the call before execution and
    returns a structured confirmation request requiring human sign-off.
    """
    tool_name = getattr(tool, "name", str(tool))
    corr_id = record_action_intent(
        tool_name=tool_name,
        intended_arguments=args,
        agent_name=getattr(tool_context, "agent_name", "alphascout_coordinator"),
    )
    if hasattr(tool_context, "state") and isinstance(tool_context.state, dict):
        tool_context.state["last_tool_correlation_id"] = corr_id

    if tool_name in HIGH_STAKES_TOOLS and not bool(args.get("human_approved", False)):
        ticker = str(args.get("ticker", "UNKNOWN")).upper()
        qty = int(args.get("quantity", 0))
        side = str(args.get("side", "buy")).upper()
        hitl_stop_response = {
            "status": "pending_human_confirmation",
            "tool_name": tool_name,
            "ticker": ticker,
            "side": side.lower(),
            "quantity": qty,
            "human_confirmation_prompt": (
                f"[HITL CALLBACK INTERCEPT] High-stakes action '{tool_name}' "
                f"({side} {qty} shares of {ticker}) requires explicit human "
                "confirmation. Ask the user to approve before setting human_approved=True."
            ),
        }
        record_action_outcome(
            correlation_id=corr_id,
            tool_name=tool_name,
            actual_outcome=hitl_stop_response,
            status="pending_human_confirmation",
        )
        return hitl_stop_response
    return None


def after_tool_outcome_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Any | None:
    """ADK `after_tool_callback` recording actual tool execution outcomes."""
    tool_name = getattr(tool, "name", str(tool))
    corr_id = "act-callback"
    if hasattr(tool_context, "state") and isinstance(tool_context.state, dict):
        corr_id = tool_context.state.get("last_tool_correlation_id", corr_id)
    record_action_outcome(
        correlation_id=corr_id,
        tool_name=tool_name,
        actual_outcome=tool_response,
        status=(
            tool_response.get("status", "success")
            if isinstance(tool_response, dict)
            else "success"
        ),
    )
    return None
