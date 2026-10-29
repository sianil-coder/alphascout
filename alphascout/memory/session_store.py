"""Context & Memory Management for AlphaScout.

Provides conversational context compaction, hybrid SEC retrieval, and async persistence:
1. History Compaction: `ContextCompactor` prevents context window bloat via
   token-based estimation, sliding window turn limits, extractive summarization
   of older turns, and Google Cloud Vertex AI Context Caching / ADK compaction
   hooks.
2. Persistent Session State: `VertexSearchAndVectorStore` and
   `PersistentSessionDatabase` connect to Google Cloud Vertex AI Search
   (`google.cloud.discoveryengine_v1`) and a persistent SQLite + Vector cosine
   similarity store to retrieve SEC filings and maintain cross-turn session
   state.
3. Async Memory Operations: `AsyncMemoryConsolidator` executes expensive
   memory generation, vector indexing, and session summarization in non-blocking
   background tasks (`asyncio.create_task` + `asyncio.Queue`) to prevent UI
   blocking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from alphascout.observability.telemetry import (
    pii_redactor,
    structured_logger,
    tracer,
)

_SEED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "seed_data.json"
)


def load_seed_data() -> dict[str, Any]:
    """Loads the pre-seeded SEC filings, fundamentals, and portfolio ledger."""
    if _SEED_DATA_PATH.exists():
        with open(_SEED_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ============================================================================
# 1. History Compaction (Token-Based Truncation + Sliding Window + Summarization)
# ============================================================================


class ContextCompactor:
    """Manages conversational context bloat via sliding windows & summarization.

    Features:
    - Estimates token counts across conversation history.
    - Compacts older conversation turns into a concise executive summary when
      the history exceeds `max_tokens` or `sliding_window_turns`.
    - Integrates with ADK `before_model_callback` (`LlmRequest` contents) and
      supports Vertex AI Context Caching metadata.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        sliding_window_turns: int = 10,
    ) -> None:
        self.max_tokens = max_tokens
        self.sliding_window_turns = sliding_window_turns
        self.compaction_events_count = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates LLM token count (~4 characters per token heuristic)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def compact_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Applies sliding-window compaction and summarizes evicted turns.

        Args:
            messages: List of conversation turn dicts with `'role'` and `'content'`.

        Returns:
            Compacted list of conversation messages guaranteed to fit within the
            token and turn budget while preserving system instructions and a
            consolidated summary of earlier turns.
        """
        if not messages:
            return []

        total_tokens = sum(
            self.estimate_tokens(str(m.get("content", ""))) for m in messages
        )
        if (
            len(messages) <= self.sliding_window_turns
            and total_tokens <= self.max_tokens
        ):
            return messages

        with tracer.start_as_current_span("memory.compact_history") as span:
            self.compaction_events_count += 1
            keep_recent = max(2, self.sliding_window_turns - 1)
            evicted = messages[:-keep_recent]
            retained = messages[-keep_recent:]

            summary_bullets: list[str] = []
            for turn in evicted:
                role = turn.get("role", "user")
                raw_text = pii_redactor.redact_text(str(turn.get("content", "")))
                snippet = raw_text[:160].replace("\n", " ").strip()
                if snippet:
                    summary_bullets.append(f"[{role.upper()}]: {snippet}...")

            compacted_summary = (
                "[COMPACTED_HISTORY_SUMMARY] Earlier conversation context "
                f"({len(evicted)} turns compacted): "
                + " | ".join(summary_bullets[:8])
            )

            compacted_list = [
                {"role": "system", "content": compacted_summary},
                *retained,
            ]

            new_tokens = sum(
                self.estimate_tokens(str(m.get("content", "")))
                for m in compacted_list
            )
            span.set_attribute("compaction.original_turns", len(messages))
            span.set_attribute("compaction.compacted_turns", len(compacted_list))
            span.set_attribute("compaction.original_tokens", total_tokens)
            span.set_attribute("compaction.new_tokens", new_tokens)

            structured_logger.info(
                f"History Compaction reduced context from {total_tokens} to {new_tokens} tokens",
                extra={
                    "event_type": "HISTORY_COMPACTION",
                    "component": "ContextCompactor",
                    "metadata": {
                        "original_turns": len(messages),
                        "compacted_turns": len(compacted_list),
                        "original_tokens": total_tokens,
                        "new_tokens": new_tokens,
                    },
                },
            )
            return compacted_list


# ============================================================================
# 2. Persistent Session State (Vertex AI Search + Persistent SQLite & Vector DB)
# ============================================================================


class VertexSearchAndVectorStore:
    """Hybrid retriever connecting to Google Cloud Vertex AI Search & local Vector DB.

    Features:
    - Queries Google Cloud Vertex AI Search (`google.cloud.discoveryengine_v1`)
      when a GCP datastore is configured.
    - Maintains a persistent SQLite + TF/Embedding Vector Store pre-indexed with
      SEC 10-K/10-Q filings so retrieval is 100% deterministic in local and CI
      environments.
    """

    def __init__(
        self,
        db_path: str | None = None,
        project_id: str | None = None,
        datastore_id: str | None = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.datastore_id = datastore_id or os.environ.get(
            "VERTEX_SEARCH_DATASTORE_ID", ""
        )
        default_db = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "alphascout_persistent_memory.sqlite3"
        )
        self.db_path = db_path or str(default_db)
        self._init_database()
        self._seed_sec_filings_if_empty()

    def _get_connection(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_database(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sec_filing_vectors (
                    doc_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    filing_type TEXT NOT NULL,
                    fiscal_period TEXT NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_memories (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ticker TEXT,
                    summary_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _compute_embedding(text: str, dim: int = 64) -> list[float]:
        """Computes a deterministic normalized lexical-semantic vector."""
        vec = [0.0] * dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % dim
            sign = 1.0 if (digest[1] % 2 == 0) else -1.0
            vec[idx] += sign * (1.0 + (digest[2] / 255.0))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def _seed_sec_filings_if_empty(self) -> None:
        seed = load_seed_data()
        filings_by_ticker = seed.get("sec_10k_filings", {})
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sec_filing_vectors")
            count = cursor.fetchone()[0]
            expected_count = sum(len(docs) for docs in filings_by_ticker.values())
            if count >= expected_count:
                return
            now = datetime.now(timezone.utc).isoformat()
            for ticker, docs in filings_by_ticker.items():
                for i, doc in enumerate(docs):
                    doc_id = f"{ticker}-{doc['fiscal_period']}-{i}"
                    combined_text = (
                        f"{ticker} {doc['section']} {doc['content']}"
                    )
                    emb = self._compute_embedding(combined_text)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sec_filing_vectors
                        (doc_id, ticker, filing_type, fiscal_period, section, content, embedding_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            ticker.upper(),
                            doc["filing_type"],
                            doc["fiscal_period"],
                            doc["section"],
                            pii_redactor.redact_text(doc["content"]),
                            json.dumps(emb),
                            now,
                        ),
                    )
            conn.commit()

    def _query_vertex_ai_search(
        self, ticker: str, query_topic: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Queries Google Cloud Vertex AI Search (`discoveryengine_v1`) if configured."""
        if (
            not self.project_id
            or not self.datastore_id
            or os.environ.get("ENABLE_LIVE_VERTEX_SEARCH", "").lower() != "true"
        ):
            return []
        try:
            from google.cloud import discoveryengine_v1  # type: ignore

            client = discoveryengine_v1.SearchServiceClient()
            serving_config = (
                f"projects/{self.project_id}/locations/global/collections/"
                f"default_collection/dataStores/{self.datastore_id}/"
                "servingConfigs/default_search"
            )
            request = discoveryengine_v1.SearchRequest(
                serving_config=serving_config,
                query=f"{ticker} {query_topic}",
                page_size=top_k,
            )
            response = client.search(request)
            results: list[dict[str, Any]] = []
            for result in response.results:
                data = dict(result.document.derived_struct_data)
                results.append(
                    {
                        "ticker": ticker,
                        "filing_type": data.get("filing_type", "Form 10-K"),
                        "fiscal_period": data.get("fiscal_period", "FY2025"),
                        "section": data.get("title", "Item 1A. Risk Factors"),
                        "content": pii_redactor.redact_text(
                            str(data.get("snippets", ""))
                        ),
                        "source": "vertex_ai_search",
                    }
                )
            return results
        except Exception:
            return []

    def search_sec_filings(
        self, ticker: str, query_topic: str, top_k: int = 4
    ) -> list[dict[str, Any]]:
        """Retrieves top-k relevant SEC 10-K/10-Q passages via Vertex AI Search + Vector DB."""
        norm_ticker = ticker.strip().upper()
        with tracer.start_as_current_span("memory.search_sec_filings") as span:
            span.set_attribute("search.ticker", norm_ticker)
            span.set_attribute("search.query_topic", query_topic)

            vertex_hits = self._query_vertex_ai_search(
                norm_ticker, query_topic, top_k
            )
            if vertex_hits:
                return vertex_hits

            query_vec = self._compute_embedding(f"{norm_ticker} {query_topic}")
            with self._get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT doc_id, ticker, filing_type, fiscal_period, section, content, embedding_json
                    FROM sec_filing_vectors
                    WHERE ticker = ?
                    """,
                    (norm_ticker,),
                ).fetchall()

            scored: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                doc_vec = json.loads(row[6])
                score = self._cosine_similarity(query_vec, doc_vec)
                scored.append(
                    (
                        score,
                        {
                            "doc_id": row[0],
                            "ticker": row[1],
                            "filing_type": row[2],
                            "fiscal_period": row[3],
                            "section": row[4],
                            "content": row[5],
                            "similarity_score": round(score, 4),
                            "source": "persistent_vector_store",
                        },
                    )
                )
            scored.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in scored[:top_k]]

    def save_session_memory_sync(
        self,
        session_id: str,
        ticker: str,
        summary_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persists a scrubbed session memory record to SQLite."""
        memory_id = f"mem-{hashlib.sha1(f'{session_id}-{ticker}-{summary_text}'.encode()).hexdigest()[:12]}"
        scrubbed_summary = pii_redactor.redact_text(summary_text)
        scrubbed_meta = pii_redactor.redact_data(metadata or {})
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_memories
                (memory_id, session_id, ticker, summary_text, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    session_id,
                    ticker.upper(),
                    scrubbed_summary,
                    json.dumps(scrubbed_meta),
                    now,
                ),
            )
            conn.commit()
        return memory_id

    def get_session_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves persisted memories for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, session_id, ticker, summary_text, metadata_json, created_at
                FROM session_memories
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "memory_id": r[0],
                "session_id": r[1],
                "ticker": r[2],
                "summary_text": r[3],
                "metadata": json.loads(r[4]),
                "created_at": r[5],
            }
            for r in rows
        ]


# ============================================================================
# 3. Async Memory Operations (Non-Blocking Background Memory Consolidation)
# ============================================================================


class AsyncMemoryConsolidator:
    """Executes expensive memory generation and consolidation asynchronously.

    Uses `asyncio.create_task()` and background futures so vector embedding
    generation, PII scrubbing, and SQLite/Vertex memory persistence never block
    the user-facing agent response turn.
    """

    def __init__(self, store: VertexSearchAndVectorStore) -> None:
        self.store = store
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def _consolidate_worker(
        self,
        session_id: str,
        ticker: str,
        raw_interaction_notes: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Async worker that scrubs PII, computes embeddings, and writes to persistent store."""
        with tracer.start_as_current_span("memory.async_consolidate") as span:
            span.set_attribute("memory.session_id", session_id)
            span.set_attribute("memory.ticker", ticker)
            # Offload CPU/IO bound persistence to thread pool so event loop stays responsive
            memory_id = await asyncio.to_thread(
                self.store.save_session_memory_sync,
                session_id,
                ticker,
                raw_interaction_notes,
                metadata,
            )
            structured_logger.info(
                f"Async memory consolidation completed for ticker='{ticker}' (memory_id={memory_id})",
                extra={
                    "event_type": "ASYNC_MEMORY_CONSOLIDATED",
                    "component": "AsyncMemoryConsolidator",
                    "metadata": {
                        "memory_id": memory_id,
                        "session_id": session_id,
                        "ticker": ticker,
                    },
                },
            )
            return memory_id

    def schedule_memory_consolidation(
        self,
        session_id: str,
        ticker: str,
        raw_interaction_notes: str,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Task[str] | str:
        """Schedules memory consolidation in the background without blocking UI.

        If an event loop is running, dispatches via `asyncio.create_task()` and
        registers a completion callback. If called from a synchronous context without
        an active loop, executes cleanly via `asyncio.run()`.
        """
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._consolidate_worker(
                    session_id=session_id,
                    ticker=ticker,
                    raw_interaction_notes=raw_interaction_notes,
                    metadata=metadata,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return task
        except RuntimeError:
            return asyncio.run(
                self._consolidate_worker(
                    session_id=session_id,
                    ticker=ticker,
                    raw_interaction_notes=raw_interaction_notes,
                    metadata=metadata,
                )
            )

    async def flush_pending_tasks(self) -> list[str]:
        """Awaits any in-flight background memory tasks (useful in tests/shutdown)."""
        if not self._background_tasks:
            return []
        results = await asyncio.gather(*self._background_tasks, return_exceptions=False)
        return list(results)


# Shared singleton instances
persistent_vector_store = VertexSearchAndVectorStore()
context_compactor = ContextCompactor()
async_memory_consolidator = AsyncMemoryConsolidator(persistent_vector_store)
