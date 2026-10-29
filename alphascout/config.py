"""Secure configuration and Google Cloud Secret Manager integration for AlphaScout.

Provides:
1. Secure Secret Management: Retrieves sensitive API keys and brokerage
   credentials dynamically via Google Cloud Secret Manager
   (`google.cloud.secretmanager_v1`) without any hardcoded secrets.
2. Strategic Model Routing: Routes high-throughput extraction and retrieval
   tasks to `gemini-2.5-flash` and complex multi-factor investment synthesis /
   coordination tasks to `gemini-2.5-pro`.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

# Model constants for Strategic Model Routing
FAST_MODEL_ID = "gemini-2.5-flash"
REASONING_MODEL_ID = "gemini-2.5-pro"

TaskComplexity = Literal[
    "fast_extraction",
    "sec_filing_retrieval",
    "insider_flow_screening",
    "adversarial_debate",
    "deep_valuation_synthesis",
    "root_orchestration",
]


class SecretManagerVault:
    """Retrieves secrets securely from Google Cloud Secret Manager.

    Never stores or uses hardcoded API keys. Connects to Google Cloud Secret
    Manager (`google.cloud.secretmanager_v1`) in cloud environments and supports
    injected runtime environment variables in CI/test sandboxes.
    """

    def __init__(self, project_id: str | None = None) -> None:
        self.project_id = project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )
        self._client = None

    def _get_client(self):
        """Lazily initializes the Google Cloud Secret Manager client."""
        if self._client is not None:
            return self._client
        try:
            from google.cloud import secretmanager_v1  # type: ignore

            self._client = secretmanager_v1.SecretManagerServiceClient()
            return self._client
        except Exception as exc:
            logger.debug(
                "Secret Manager client initialization deferred (%s); "
                "using runtime environment injection.",
                exc,
            )
            return None

    @functools.lru_cache(maxsize=32)
    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        env_fallback_var: str | None = None,
    ) -> str | None:
        """Fetches a secret payload from Google Cloud Secret Manager.

        Args:
            secret_id: The ID of the secret in Google Cloud Secret Manager.
            version: The secret version to access (default: "latest").
            env_fallback_var: Optional environment variable name used when
                running in isolated CI/CD runners without GCP credentials.

        Returns:
            The decoded secret string, or None if unavailable.
        """
        if env_fallback_var and os.environ.get(env_fallback_var):
            return os.environ.get(env_fallback_var)

        client = self._get_client()
        if client is not None and self.project_id:
            resource_name = (
                f"projects/{self.project_id}/secrets/{secret_id}"
                f"/versions/{version}"
            )
            try:
                response = client.access_secret_version(
                    request={"name": resource_name}
                )
                payload = response.payload.data.decode("UTF-8").strip()
                logger.info(
                    "Successfully resolved secret '%s' from Secret Manager.",
                    secret_id,
                )
                return payload
            except Exception as exc:
                logger.warning(
                    "Could not fetch secret '%s' from Secret Manager (%s). "
                    "Falling back to sandbox simulation mode.",
                    secret_id,
                    type(exc).__name__,
                )
        return None


@dataclasses.dataclass(frozen=True)
class AlphaScoutConfig:
    """Central configuration for the AlphaScout Multi-Agent system."""

    project_id: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "alphascout-gcp-project"
        )
    )
    location: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )
    )
    vertex_search_datastore_id: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "VERTEX_SEARCH_DATASTORE_ID", "alphascout-sec-filings-datastore"
        )
    )
    alpaca_paper_base_url: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"
        )
    )
    alpaca_api_key_secret_id: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "ALPACA_API_KEY_SECRET_ID", "alphascout-alpaca-paper-api-key"
        )
    )
    alpaca_secret_key_secret_id: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "ALPACA_SECRET_KEY_SECRET_ID", "alphascout-alpaca-paper-secret-key"
        )
    )
    max_context_tokens: int = dataclasses.field(
        default_factory=lambda: int(
            os.environ.get("MAX_CONTEXT_TOKENS", "8000")
        )
    )
    sliding_window_turns: int = dataclasses.field(
        default_factory=lambda: int(
            os.environ.get("SLIDING_WINDOW_TURNS", "10")
        )
    )
    fast_model: str = FAST_MODEL_ID
    reasoning_model: str = REASONING_MODEL_ID


def route_model_by_task_complexity(
    task_type: TaskComplexity,
    config: AlphaScoutConfig | None = None,
) -> str:
    """Strategically routes agent workloads to the optimal Gemini model tier.

    Fast data extraction, SEC filing chunk retrieval, insider trade screening,
    and initial Bull/Bear debate arguments use `gemini-2.5-flash` for low
    latency and cost efficiency. High-reasoning investment synthesis and root
    multi-agent orchestration use `gemini-2.5-pro`.

    Args:
        task_type: The classification of the sub-agent or task workload.
        config: Optional AlphaScoutConfig instance.

    Returns:
        The Gemini model identifier string (e.g., 'gemini-2.5-flash' or
        'gemini-2.5-pro').
    """
    cfg = config or AlphaScoutConfig()
    routing_table: dict[TaskComplexity, str] = {
        "fast_extraction": cfg.fast_model,
        "sec_filing_retrieval": cfg.fast_model,
        "insider_flow_screening": cfg.fast_model,
        "adversarial_debate": cfg.fast_model,
        "deep_valuation_synthesis": cfg.reasoning_model,
        "root_orchestration": cfg.reasoning_model,
    }
    selected_model = routing_table.get(task_type, cfg.fast_model)
    logger.debug(
        "Strategic Model Router assigned model='%s' for task_type='%s'",
        selected_model,
        task_type,
    )
    return selected_model
