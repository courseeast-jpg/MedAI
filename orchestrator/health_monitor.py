"""
MedAI v1.1 — Claude Health Monitor
Background task: pings Claude API every 60s.
Triggers safe mode on failure. Auto-recovers.
Processes pending enrichment queue on recovery.
"""
import asyncio
import time
from loguru import logger

import anthropic

from app.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    CLAUDE_HEALTH_CHECK_INTERVAL_SEC, CLAUDE_TIMEOUT_SEC
)
from app.schemas import SystemState, LedgerEvent


class ClaudeHealthMonitor:
    def __init__(self, state: SystemState, sql_store=None, enrichment_engine=None):
        self.state = state
        self.sql = sql_store
        self.enrichment = enrichment_engine
        self._running = False
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    async def run(self):
        """Background loop. Run as asyncio task."""
        self._running = True
        logger.info("Claude health monitor started")
        while self._running:
            await asyncio.sleep(CLAUDE_HEALTH_CHECK_INTERVAL_SEC)
            await self._check()

    async def _check(self):
        if not self._client:
            self._set_unavailable("No API key configured")
            return

        try:
            # Lightweight ping — minimal tokens
            self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
                timeout=CLAUDE_TIMEOUT_SEC,
            )
            if not self.state.claude_available:
                await self._on_recovery()
            else:
                self.state.claude_available = True

        except anthropic.RateLimitError:
            logger.warning("Claude rate limited — maintaining current state")
            # Don't trigger safe mode for rate limits — temporary condition

        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            logger.warning(f"Claude health check failed: {e}")
            self._set_unavailable(str(e)[:100])

        except Exception as e:
            logger.error(f"Unexpected health check error: {e}")
            self._set_unavailable(str(e)[:100])

    def _set_unavailable(self, reason: str):
        was_available = self.state.claude_available
        self.state.claude_available = False
        self.state.safe_mode = True
        self.state.safe_mode_reason = reason

        if was_available:
            logger.error(f"Claude UNAVAILABLE — entering safe mode. Reason: {reason}")
            if self.sql:
                self.sql.write_ledger(LedgerEvent(
                    event_type="claude_unavailable",
                    details={"reason": reason},
                ))

    async def _on_recovery(self):
        logger.info("Claude API recovered — exiting safe mode")
        self.state.claude_available = True
        self.state.safe_mode = False
        self.state.safe_mode_reason = None

        if self.sql:
            self.sql.write_ledger(LedgerEvent(
                event_type="claude_recovered",
                details={"action": "safe_mode_exited"},
            ))

        # Process pending enrichment queue
        if self.enrichment:
            processed = self.enrichment.process_pending_queue()
            if processed > 0:
                logger.info(f"Processed {processed} queued enrichment items after Claude recovery")

    def stop(self):
        self._running = False

    @property
    def is_healthy(self) -> bool:
        return self.state.claude_available
