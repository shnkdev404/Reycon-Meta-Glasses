"""
Phase 9: Directed Alert Decision Engine.

Enforces non-broadcast directed spatial alert routing.
Ensures warnings are sent EXCLUSIVELY to affected smart glass units (e.g. Glass B),
preventing unnecessary noise for unaffected glasses (e.g. Glass A).
"""
from typing import List
from app.models.threat import ThreatAlert
from app.services.connection_manager import connection_manager
from app.meta.alert_adapter import MetaAlertAdapter
from app.utils.logger import get_logger

logger = get_logger("AlertEngine")


import time
from typing import List, Dict, Set
from datetime import datetime, timezone
from app.models.threat import ThreatAlert
from app.services.connection_manager import connection_manager
from app.meta.alert_adapter import MetaAlertAdapter
from app.utils.logger import get_logger

logger = get_logger("AlertEngine")


class AlertDecisionEngine:
    """Filters, deduplicates, and dispatches non-broadcast directed threat alerts."""

    def __init__(self, throttle_seconds: float = 1.0):
        self.throttle_seconds = throttle_seconds
        self._sent_alert_records: Dict[str, float] = {}
        self._history: List[Dict] = []

    async def dispatch_alerts(self, alerts: List[ThreatAlert]) -> int:
        """
        Process active threat alerts and send targeted warnings directly
        to affected smart glasses over WebSocket and Meta Glass HUD adapter.
        """
        dispatched_count = 0
        now_ts = time.time()

        for alert in alerts:
            # Deduplicate recently dispatched alerts to prevent spamming
            dedup_key = f"{alert.target_glass_id}_{alert.trigger_object_id}_{alert.threat_level.value}"
            last_sent = self._sent_alert_records.get(dedup_key, 0.0)

            if (now_ts - last_sent) < self.throttle_seconds:
                continue

            self._sent_alert_records[dedup_key] = now_ts

            # Construct targeted WebSocket JSON alert payload
            payload = {
                "type": "THREAT_ALERT",
                "alert": alert.model_dump(mode="json")
            }

            # 1. Send EXCLUSIVELY over WebSocket to target glass connection (Non-broadcast)
            ws_success = await connection_manager.send_direct_message(
                glass_id=alert.target_glass_id,
                message=payload
            )

            # 2. Trigger Meta Smart Glass HUD / Audio / Haptic adapter
            meta_adapter = MetaAlertAdapter(glass_id=alert.target_glass_id)
            await meta_adapter.send_alert(alert)

            # 3. Log alert dispatch history audit record
            self._history.append({
                "alert_id": alert.alert_id,
                "target_glass_id": alert.target_glass_id,
                "trigger_object_id": alert.trigger_object_id,
                "threat_level": alert.threat_level.value,
                "threat_type": alert.threat_type.value,
                "warning_message": alert.warning_message,
                "websocket_delivered": ws_success,
                "dispatched_at": datetime.now(timezone.utc).isoformat()
            })

            # Bound history size
            if len(self._history) > 200:
                self._history.pop(0)

            dispatched_count += 1
            logger.info(
                f"📡 Directed alert dispatched ONLY to target glass '{alert.target_glass_id}' "
                f"(WebSocket status: {ws_success})"
            )

        # Cleanup old throttling records
        if len(self._sent_alert_records) > 1000:
            self._sent_alert_records.clear()

        return dispatched_count

    def get_alert_history(self, glass_id: str = None) -> List[Dict]:
        """Retrieve audit history of dispatched alerts."""
        if glass_id:
            return [a for a in self._history if a["target_glass_id"] == glass_id]
        return list(self._history)

    def clear_alert_history(self):
        """Clear alert history log and throttling records."""
        self._history.clear()
        self._sent_alert_records.clear()


alert_engine = AlertDecisionEngine()

