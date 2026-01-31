"""
Alert System for Self-Healing Events.

Provides multi-channel alerting with severity levels,
deduplication, and template-based messaging.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json

from .config import AlertSystemConfig
from .exceptions import AlertError


logger = logging.getLogger(__name__)


AlertSeverity = Literal["INFO", "WARNING", "CRITICAL"]


class AlertSystem:
    """
    Multi-channel alert system for self-healing events.
    
    Features:
    - Multiple channels (Slack, Email, Webhook)
    - Severity-based routing
    - Alert deduplication and throttling
    - Template-based messages
    - Alert history tracking
    
    Args:
        config: AlertSystemConfig instance
    """
    
    def __init__(self, config: Optional[AlertSystemConfig] = None):
        if config is None:
            config = AlertSystemConfig()
        
        self.config = config
        
        # Alert deduplication tracking
        self.recent_alerts: Dict[str, datetime] = {}
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        
        # Setup channels
        self.channels = self._setup_channels()
        
        logger.info(
            "AlertSystem initialized with channels: %s, min_severity=%s",
            list(self.channels.keys()), config.min_severity_level
        )
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = "WARNING",
        context: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Send an alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            context: Optional context dictionary
            force: Force send even if deduplicated
        
        Returns:
            Alert result dictionary
        """
        try:
            timestamp = datetime.now()
            
            # Check severity filter
            if not self._should_send_severity(severity):
                logger.debug("Alert suppressed due to severity filter: %s < %s",
                           severity, self.config.min_severity_level)
                return {
                    'sent': False,
                    'reason': 'severity_filtered',
                    'severity': severity
                }
            
            # Check deduplication
            if not force and self.config.enable_deduplication:
                if self._is_duplicate(title, message):
                    logger.debug("Alert deduplicated: %s", title)
                    return {
                        'sent': False,
                        'reason': 'deduplicated',
                        'title': title
                    }
            
            # Format message
            formatted_message = self._format_message(title, message, severity, context)
            
            # Send through channels
            results = {}
            for channel_name, channel in self.channels.items():
                try:
                    channel_result = channel(formatted_message, severity)
                    results[channel_name] = {
                        'success': True,
                        'result': channel_result
                    }
                except Exception as e:
                    logger.error("Failed to send alert via %s: %s", channel_name, str(e))
                    results[channel_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Record alert
            alert_record = {
                'timestamp': timestamp.isoformat(),
                'title': title,
                'message': message,
                'severity': severity,
                'context': context,
                'channels_used': list(self.channels.keys()),
                'results': results,
            }
            
            self.alert_history.append(alert_record)
            
            # Update deduplication tracking
            self._mark_sent(title, message, timestamp)
            
            logger.info(
                "Alert sent: severity=%s, title='%s', channels=%s",
                severity, title, list(results.keys())
            )
            
            return {
                'sent': True,
                'timestamp': timestamp.isoformat(),
                'channels': results,
                'severity': severity
            }
            
        except Exception as e:
            logger.error("Failed to send alert: %s", str(e), exc_info=True)
            raise AlertError(f"Failed to send alert: {e}") from e
    
    def _setup_channels(self) -> Dict[str, callable]:
        """Setup active alert channels."""
        channels = {}
        
        if self.config.enable_slack:
            channels['slack'] = self._send_slack
        
        if self.config.enable_email:
            channels['email'] = self._send_email
        
        if self.config.enable_webhook:
            channels['webhook'] = self._send_webhook
        
        # Always have console channel for testing
        if not channels:
            logger.warning("No alert channels configured, using console only")
            channels['console'] = self._send_console
        
        return channels
    
    def _should_send_severity(self, severity: AlertSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        severity_order = ["INFO", "WARNING", "CRITICAL"]
        min_level = severity_order.index(self.config.min_severity_level)
        current_level = severity_order.index(severity)
        return current_level >= min_level
    
    def _is_duplicate(self, title: str, message: str) -> bool:
        """Check if this alert is a duplicate."""
        key = f"{title}:{message}"
        
        if key in self.recent_alerts:
            last_sent = self.recent_alerts[key]
            time_diff = datetime.now() - last_sent
            dedup_window = timedelta(minutes=self.config.dedup_window_minutes)
            
            if time_diff < dedup_window:
                return True
        
        return False
    
    def _mark_sent(self, title: str, message: str, timestamp: datetime) -> None:
        """Mark an alert as sent for deduplication."""
        key = f"{title}:{message}"
        self.recent_alerts[key] = timestamp
        
        # Cleanup old entries
        cutoff = timestamp - timedelta(minutes=self.config.dedup_window_minutes * 2)
        self.recent_alerts = {
            k: v for k, v in self.recent_alerts.items()
            if v > cutoff
        }
    
    def _format_message(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format alert message with context."""
        parts = [
            f"🚨 **{severity}** Alert",
            f"**{title}**",
            "",
            message,
        ]
        
        if context:
            parts.append("")
            parts.append("**Context:**")
            for key, value in context.items():
                if isinstance(value, dict):
                    value_str = json.dumps(value, indent=2)
                else:
                    value_str = str(value)
                parts.append(f"- {key}: {value_str}")
        
        parts.append("")
        parts.append(f"_Timestamp: {datetime.now().isoformat()}_")
        
        return "\n".join(parts)
    
    def _send_slack(self, message: str, severity: AlertSeverity) -> str:
        """Send alert via Slack."""
        if not self.config.slack_webhook_url:
            raise AlertError("Slack webhook URL not configured")
        
        try:
            import requests
            
            # Color based on severity
            colors = {
                "INFO": "#36a64f",
                "WARNING": "#ff9900",
                "CRITICAL": "#ff0000"
            }
            
            payload = {
                "channel": self.config.slack_channel,
                "attachments": [{
                    "color": colors.get(severity, "#808080"),
                    "text": message,
                    "footer": "Self-Healing System",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            return f"Slack notification sent (status={response.status_code})"
            
        except ImportError:
            raise AlertError("requests library not installed for Slack integration")
        except Exception as e:
            raise AlertError(f"Slack send failed: {e}") from e
    
    def _send_email(self, message: str, severity: AlertSeverity) -> str:
        """Send alert via email."""
        if not self.config.email_recipients:
            raise AlertError("Email recipients not configured")
        
        logger.warning("Email alerts not fully implemented - would send to: %s",
                      self.config.email_recipients)
        return f"Email would be sent to {len(self.config.email_recipients)} recipients"
    
    def _send_webhook(self, message: str, severity: AlertSeverity) -> str:
        """Send alert via webhook."""
        if not self.config.webhook_url:
            raise AlertError("Webhook URL not configured")
        
        try:
            import requests
            
            payload = {
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "source": "self_healing_system"
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            return f"Webhook sent (status={response.status_code})"
            
        except ImportError:
            raise AlertError("requests library not installed for webhook integration")
        except Exception as e:
            raise AlertError(f"Webhook send failed: {e}") from e
    
    def _send_console(self, message: str, severity: AlertSeverity) -> str:
        """Send alert to console (fallback/testing)."""
        log_levels = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL
        }
        
        logger.log(log_levels.get(severity, logging.INFO), "ALERT:\n%s", message)
        return "Console alert logged"
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert history."""
        if not self.alert_history:
            return {'alerts_sent': 0}
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert['severity']] += 1
        
        return {
            'alerts_sent': len(self.alert_history),
            'by_severity': dict(severity_counts),
            'recent_alerts': self.alert_history[-10:],
            'active_dedup_keys': len(self.recent_alerts),
        }
