"""
Event Bus - Inter-agent communication
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

from typing import Dict, Any, Callable, List
from datetime import datetime
import logging


class EventBus:
    """
    Simple event bus for inter-agent communication.
    Allows agents to publish and subscribe to events.
    """
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger("ALC.EventBus")
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type}")
    
    def publish(self, event_type: str, data: Dict[str, Any]):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.logger.info(f"Publishing event: {event_type}")
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event
            callback: Callback to remove
        """
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                cb for cb in self.subscribers[event_type] if cb != callback
            ]
            self.logger.debug(f"Unsubscribed from {event_type}")


# Global event bus instance
event_bus = EventBus()

