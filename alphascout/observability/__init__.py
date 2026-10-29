"""Observability package for AlphaScout."""

from alphascout.observability.telemetry import (
    PIIRedactionEngine,
    clear_audit_log_buffer,
    clear_exported_spans,
    get_audit_log_buffer,
    get_exported_spans,
    get_structured_logger,
    init_tracer,
    pii_redactor,
    record_action_intent,
    record_action_outcome,
    structured_logger,
    traced_tool_execution,
    tracer,
)

__all__ = [
    "PIIRedactionEngine",
    "clear_audit_log_buffer",
    "clear_exported_spans",
    "get_audit_log_buffer",
    "get_exported_spans",
    "get_structured_logger",
    "init_tracer",
    "pii_redactor",
    "record_action_intent",
    "record_action_outcome",
    "structured_logger",
    "traced_tool_execution",
    "tracer",
]
