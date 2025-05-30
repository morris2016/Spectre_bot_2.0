import asyncio
from typing import Any, Callable, Dict, List

from .logger import get_logger


class EventBus:
    """Asynchronous publish/subscribe event bus."""

    _instance: "EventBus" = None

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Any], Any]]] = {}
        self._lock = asyncio.Lock()
        self.logger = get_logger("EventBus")

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def subscribe(self, event: str, callback: Callable[[Any], Any]) -> None:
        async with self._lock:
            self._subscribers.setdefault(event, []).append(callback)
            self.logger.debug(f"Subscribed to {event}")

    async def unsubscribe(self, event: str, callback: Callable[[Any], Any]) -> None:
        async with self._lock:
            callbacks = self._subscribers.get(event, [])
            if callback in callbacks:
                callbacks.remove(callback)
                self.logger.debug(f"Unsubscribed from {event}")
            if not callbacks and event in self._subscribers:
                del self._subscribers[event]

    async def publish(self, event: str, data: Any) -> None:
        async with self._lock:
            callbacks = list(self._subscribers.get(event, []))
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(data))
                else:
                    cb(data)
            except Exception as exc:
                self.logger.error(f"Error in event handler for {event}: {exc}")
