"""Metrics service for tracking performance."""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TTFAMetric:
    """Time to First Audio metric."""
    message_id: str
    sender_id: str
    recipient_id: str
    source_lang: str
    target_lang: str
    client_sent_at: datetime
    server_received_at: datetime
    translation_completed_at: Optional[datetime] = None
    ws_sent_at: Optional[datetime] = None
    client_played_at: Optional[datetime] = None
    ttfa_ms: Optional[int] = None


class MetricsService:
    """Service for tracking and analyzing performance metrics."""
    
    def __init__(self) -> None:
        self.ttfa_metrics: Dict[str, TTFAMetric] = {}
        self.translation_times: List[float] = []
        
    def start_ttfa_tracking(
        self,
        message_id: str,
        sender_id: str,
        recipient_id: str,
        source_lang: str,
        target_lang: str,
        client_sent_at: datetime
    ) -> None:
        """Start tracking TTFA for a message."""
        metric = TTFAMetric(
            message_id=message_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            source_lang=source_lang,
            target_lang=target_lang,
            client_sent_at=client_sent_at,
            server_received_at=datetime.utcnow()
        )
        self.ttfa_metrics[message_id] = metric
        logger.debug(f"Started TTFA tracking for message {message_id}")
    
    def record_translation_completed(self, message_id: str) -> None:
        """Record when translation is completed."""
        if message_id in self.ttfa_metrics:
            metric = self.ttfa_metrics[message_id]
            metric.translation_completed_at = datetime.utcnow()
            
            # Calculate translation time
            if metric.server_received_at:
                translation_time = (
                    metric.translation_completed_at - metric.server_received_at
                ).total_seconds()
                self.translation_times.append(translation_time)
                logger.debug(f"Translation completed for {message_id} in {translation_time:.3f}s")
    
    def record_ws_sent(self, message_id: str) -> None:
        """Record when message is sent via WebSocket."""
        if message_id in self.ttfa_metrics:
            self.ttfa_metrics[message_id].ws_sent_at = datetime.utcnow()
            logger.debug(f"Message {message_id} sent via WebSocket")
    
    def record_client_played(self, message_id: str) -> int:
        """
        Record when client starts playing audio and calculate TTFA.
        
        Returns:
            TTFA in milliseconds
        """
        if message_id not in self.ttfa_metrics:
            return 0
            
        metric = self.ttfa_metrics[message_id]
        metric.client_played_at = datetime.utcnow()
        
        # Calculate TTFA (client sent to client played)
        if metric.client_sent_at:
            ttfa = (metric.client_played_at - metric.client_sent_at).total_seconds() * 1000
            metric.ttfa_ms = int(ttfa)
            
            logger.info(
                f"TTFA for message {message_id}: {metric.ttfa_ms}ms "
                f"({metric.source_lang} -> {metric.target_lang})"
            )
            
            return metric.ttfa_ms
        
        return 0
    
    def get_ttfa_stats(self) -> Dict[str, Any]:
        """Get TTFA statistics."""
        completed_ttfas = [
            metric.ttfa_ms for metric in self.ttfa_metrics.values() 
            if metric.ttfa_ms is not None
        ]
        
        if not completed_ttfas:
            return {
                "count": 0,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "p95_ms": 0
            }
        
        completed_ttfas.sort()
        count = len(completed_ttfas)
        avg_ms = sum(completed_ttfas) / count
        min_ms = completed_ttfas[0]
        max_ms = completed_ttfas[-1]
        p95_index = int(0.95 * count)
        p95_ms = completed_ttfas[p95_index] if p95_index < count else max_ms
        
        return {
            "count": count,
            "avg_ms": round(avg_ms, 2),
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p95_ms": p95_ms
        }
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation time statistics."""
        if not self.translation_times:
            return {
                "count": 0,
                "avg_seconds": 0,
                "min_seconds": 0,
                "max_seconds": 0,
                "p95_seconds": 0
            }
        
        times_sorted = sorted(self.translation_times)
        count = len(times_sorted)
        avg_seconds = sum(times_sorted) / count
        min_seconds = times_sorted[0]
        max_seconds = times_sorted[-1]
        p95_index = int(0.95 * count)
        p95_seconds = times_sorted[p95_index] if p95_index < count else max_seconds
        
        return {
            "count": count,
            "avg_seconds": round(avg_seconds, 3),
            "min_seconds": round(min_seconds, 3),
            "max_seconds": round(max_seconds, 3),
            "p95_seconds": round(p95_seconds, 3)
        }


# Global metrics service instance
metrics_service = MetricsService()
