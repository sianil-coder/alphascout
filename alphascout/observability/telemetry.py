"""Observability, OpenTelemetry Distributed Tracing, Intent/Outcome Audit, and PII Redaction.

Provides telemetry, privacy redaction, and distributed tracing primitives:
1. Structured JSON Logging: Outputs machine-parseable JSON logs compatible
   with Google Cloud Logging (`severity`, `timestamp`, `trace_id`, `span_id`,
   `event_type`, `metadata`).
2. Intent vs. Outcome Capture: Explicitly logs `AGENT_ACTION_INTENT` before
   tool/model execution and `AGENT_ACTION_OUTCOME` after execution with latency
   and correlation IDs.
3. Distributed Tracing: Configures OpenTelemetry (`TracerProvider`, spans)
   to link spans from user query -> coordinator -> sub-agents -> tool calls -> final answer.
4. PII Redaction: Scrubs sensitive PII (SSNs, brokerage account IDs, emails,
   phone numbers, credit card numbers, API keys) before writing to logs or memory,
   integrating Google Cloud DLP API (`google.cloud.dlp_v2`) with a deterministic
   regex scrubber fallback.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Any, Generator
import uuid

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore
        InMemorySpanExporter,
    )

    _HAS_OTEL_SDK = True
except ImportError:
    _HAS_OTEL_SDK = False

    class _FallbackSpanContext:
        is_valid = True

        def __init__(self) -> None:
            self.trace_id = uuid.uuid4().int & ((1 << 128) - 1)
            self.span_id = uuid.uuid4().int & ((1 << 64) - 1)

    class _FallbackSpan:
        def __init__(self, name: str, exporter: "_FallbackInMemoryExporter") -> None:
            self.name = name
            self.attributes: dict[str, Any] = {}
            self.exceptions: list[Exception] = []
            self._context = _FallbackSpanContext()
            self._exporter = exporter

        def __enter__(self) -> "_FallbackSpan":
            _FallbackTraceModule._current_span = self
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self._exporter.record(self)
            _FallbackTraceModule._current_span = None

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

        def record_exception(self, exc: Exception) -> None:
            self.exceptions.append(exc)

        def get_span_context(self) -> _FallbackSpanContext:
            return self._context

    class _FallbackInMemoryExporter:
        def __init__(self) -> None:
            self._spans: list[_FallbackSpan] = []

        def record(self, span: _FallbackSpan) -> None:
            self._spans.append(span)

        def get_finished_spans(self) -> tuple[_FallbackSpan, ...]:
            return tuple(self._spans)

        def clear(self) -> None:
            self._spans.clear()

    class _FallbackTracer:
        def __init__(self, exporter: _FallbackInMemoryExporter) -> None:
            self._exporter = exporter

        def start_as_current_span(self, name: str) -> _FallbackSpan:
            return _FallbackSpan(name, self._exporter)

    class _FallbackTraceModule:
        _current_span: _FallbackSpan | None = None

        @classmethod
        def get_current_span(cls) -> _FallbackSpan | None:
            return cls._current_span

        @staticmethod
        def set_tracer_provider(provider: Any) -> None:
            pass

        @staticmethod
        def get_tracer(name: str) -> _FallbackTracer:
            return _FallbackTracer(_IN_MEMORY_SPAN_EXPORTER)  # type: ignore

    trace = _FallbackTraceModule()  # type: ignore
    InMemorySpanExporter = _FallbackInMemoryExporter  # type: ignore

# ============================================================================
# 1. PII Redaction Engine (Cloud DLP API + Deterministic Regex Scrubber)
# ============================================================================


class PIIRedactionEngine:
    """Scrubs sensitive PII/financial identifiers from logs, traces, and memory.

    Integrates with Google Cloud Sensitive Data Protection (`google.cloud.dlp_v2`)
    when enabled in GCP and always runs a deterministic high-speed regex scrubber
    so zero PII ever reaches disk, logs, or vector/session storage.
    """

    _PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # US Social Security Numbers (XXX-XX-XXXX)
        (
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "[REDACTED_SSN]",
        ),
        # Client Brokerage Account Identifiers (e.g., ACC-8841-9920, ACCT-12345678)
        (
            re.compile(r"\bACC(?:T)?-[A-Z0-9\-]{4,20}\b", re.IGNORECASE),
            "[REDACTED_ACCOUNT_ID]",
        ),
        # Email Addresses
        (
            re.compile(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
            ),
            "[REDACTED_EMAIL]",
        ),
        # Credit / Debit Card Numbers (4 groups of 4 digits separated by space or dash)
        (
            re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
            "[REDACTED_CARD_NUMBER]",
        ),
        # US Phone Numbers
        (
            re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
            ),
            "[REDACTED_PHONE]",
        ),
        # Bearer / API Secret Tokens
        (
            re.compile(
                r"\b(?:sk-[A-Za-z0-9]{16,}|PK[A-Z0-9]{14,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*)\b"
            ),
            "[REDACTED_API_KEY]",
        ),
    ]

    def __init__(
        self,
        project_id: str | None = None,
        use_cloud_dlp: bool = False,
    ) -> None:
        self.project_id = project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )
        self.use_cloud_dlp = use_cloud_dlp
        self._dlp_client = None

    def _deidentify_with_cloud_dlp(self, text: str) -> str:
        """Calls Google Cloud DLP API (`google.cloud.dlp_v2`) if enabled."""
        if not self.use_cloud_dlp or not self.project_id:
            return text
        try:
            from google.cloud import dlp_v2  # type: ignore

            if self._dlp_client is None:
                self._dlp_client = dlp_v2.DlpServiceClient()
            parent = f"projects/{self.project_id}/locations/global"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "FINANCIAL_ACCOUNT_NUMBER"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = self._dlp_client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": text},
                }
            )
            return response.item.value
        except Exception:
            return text

    def redact_text(self, text: str) -> str:
        """Scrubs all PII tokens from a raw string."""
        if not text or not isinstance(text, str):
            return text
        scrubbed = self._deidentify_with_cloud_dlp(text)
        for pattern, replacement in self._PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    def redact_data(self, payload: Any) -> Any:
        """Recursively scrubs PII across dicts, lists, and strings."""
        if isinstance(payload, str):
            return self.redact_text(payload)
        if isinstance(payload, dict):
            return {k: self.redact_data(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self.redact_data(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.redact_data(item) for item in payload)
        return payload


# Global singleton PII Redactor
pii_redactor = PIIRedactionEngine()


# ============================================================================
# 2. Distributed Tracing with OpenTelemetry
# ============================================================================

_IN_MEMORY_SPAN_EXPORTER = InMemorySpanExporter()
_TRACER_INITIALIZED = False


def init_tracer(service_name: str = "alphascout-equity-research-agent") -> Any:
    """Initializes OpenTelemetry TracerProvider with GCP & In-Memory exporters."""
    global _TRACER_INITIALIZED
    if not _TRACER_INITIALIZED and _HAS_OTEL_SDK:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "1.0.0",
                "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            SimpleSpanProcessor(_IN_MEMORY_SPAN_EXPORTER)
        )
        if os.environ.get("ENABLE_OTEL_CONSOLE_EXPORT", "").lower() == "true":
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
        if os.environ.get("ENABLE_CLOUD_TRACE_EXPORT", "").lower() == "true":
            try:
                from opentelemetry.exporter.cloud_trace import (  # type: ignore
                    CloudTraceSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(CloudTraceSpanExporter())
                )
            except Exception:
                pass

        trace.set_tracer_provider(provider)
        _TRACER_INITIALIZED = True
    else:
        _TRACER_INITIALIZED = True
    return trace.get_tracer(service_name)


def get_exported_spans():
    """Returns recorded OpenTelemetry spans for verification and evaluation."""
    return _IN_MEMORY_SPAN_EXPORTER.get_finished_spans()


def clear_exported_spans() -> None:
    """Clears the in-memory OpenTelemetry span buffer."""
    _IN_MEMORY_SPAN_EXPORTER.clear()


tracer = init_tracer()


# ============================================================================
# 3. Structured JSON Logging with Automatic PII Scrubbing
# ============================================================================


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with OpenTelemetry trace context."""

    def format(self, record: logging.LogRecord) -> str:
        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span else None
        trace_id = (
            format(span_ctx.trace_id, "032x")
            if span_ctx and span_ctx.is_valid
            else getattr(record, "trace_id", "0" * 32)
        )
        span_id = (
            format(span_ctx.span_id, "016x")
            if span_ctx and span_ctx.is_valid
            else getattr(record, "span_id", "0" * 16)
        )

        raw_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "logging.googleapis.com/trace": trace_id,
            "logging.googleapis.com/spanId": span_id,
            "event_type": getattr(record, "event_type", "APPLICATION_LOG"),
            "component": getattr(record, "component", "alphascout"),
            "metadata": getattr(record, "metadata", {}),
        }
        scrubbed_payload = pii_redactor.redact_data(raw_payload)
        return json.dumps(scrubbed_payload, default=str)


def get_structured_logger(name: str = "alphascout") -> logging.Logger:
    """Returns a structured JSON logger with automatic PII redaction."""
    struct_logger = logging.getLogger(name)
    if not struct_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        struct_logger.addHandler(handler)
        struct_logger.setLevel(logging.INFO)
        struct_logger.propagate = False
    return struct_logger


structured_logger = get_structured_logger("alphascout.telemetry")


# ============================================================================
# 4. Intent vs. Outcome Capture (Audit Trail for Tool & Agent Executions)
# ============================================================================

_AUDIT_LOG_BUFFER: list[dict[str, Any]] = []


def get_audit_log_buffer() -> list[dict[str, Any]]:
    """Returns captured Intent vs. Outcome structured audit events."""
    return list(_AUDIT_LOG_BUFFER)


def clear_audit_log_buffer() -> None:
    """Clears the in-memory audit event buffer."""
    _AUDIT_LOG_BUFFER.clear()


def record_action_intent(
    tool_name: str,
    intended_arguments: dict[str, Any],
    agent_name: str = "alphascout_coordinator",
    correlation_id: str | None = None,
) -> str:
    """Logs the agent's INTENDED action and arguments BEFORE execution."""
    corr_id = correlation_id or f"act-{uuid.uuid4().hex[:12]}"
    scrubbed_args = pii_redactor.redact_data(intended_arguments)
    event = {
        "event_type": "AGENT_ACTION_INTENT",
        "correlation_id": corr_id,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "intended_arguments": scrubbed_args,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _AUDIT_LOG_BUFFER.append(event)
    structured_logger.info(
        f"INTENT: Agent '{agent_name}' intends to invoke tool '{tool_name}'",
        extra={
            "event_type": "AGENT_ACTION_INTENT",
            "component": agent_name,
            "metadata": event,
        },
    )
    return corr_id


def record_action_outcome(
    correlation_id: str,
    tool_name: str,
    actual_outcome: Any,
    status: str = "success",
    latency_ms: float = 0.0,
    agent_name: str = "alphascout_coordinator",
) -> dict[str, Any]:
    """Logs the agent's ACTUAL outcome AFTER tool execution."""
    scrubbed_outcome = pii_redactor.redact_data(actual_outcome)
    event = {
        "event_type": "AGENT_ACTION_OUTCOME",
        "correlation_id": correlation_id,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "actual_outcome": scrubbed_outcome,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _AUDIT_LOG_BUFFER.append(event)
    structured_logger.info(
        f"OUTCOME: Tool '{tool_name}' completed with status='{status}' in {latency_ms:.2f}ms",
        extra={
            "event_type": "AGENT_ACTION_OUTCOME",
            "component": agent_name,
            "metadata": event,
        },
    )
    return event


@contextlib.contextmanager
def traced_tool_execution(
    tool_name: str,
    arguments: dict[str, Any],
    agent_name: str = "alphascout_coordinator",
) -> Generator[dict[str, Any], None, None]:
    """Context manager combining OpenTelemetry span + Intent/Outcome audit logging."""
    start_time = time.perf_counter()
    scrubbed_args = pii_redactor.redact_data(arguments)
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute(
            "tool.arguments_json", json.dumps(scrubbed_args, default=str)
        )
        corr_id = record_action_intent(
            tool_name=tool_name,
            intended_arguments=scrubbed_args,
            agent_name=agent_name,
        )
        state: dict[str, Any] = {
            "correlation_id": corr_id,
            "outcome": None,
            "status": "success",
        }
        try:
            yield state
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            span.set_attribute("tool.status", state["status"])
            span.set_attribute("tool.latency_ms", round(latency_ms, 2))
            record_action_outcome(
                correlation_id=corr_id,
                tool_name=tool_name,
                actual_outcome=state["outcome"],
                status=state["status"],
                latency_ms=latency_ms,
                agent_name=agent_name,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            span.record_exception(exc)
            span.set_attribute("tool.status", "error")
            record_action_outcome(
                correlation_id=corr_id,
                tool_name=tool_name,
                actual_outcome={"error": str(exc)},
                status="error",
                latency_ms=latency_ms,
                agent_name=agent_name,
            )
            raise
