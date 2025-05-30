#!/usr/bin/env python3
"""Base classes for hardware accelerators."""
from typing import Dict, Any

from common.logger import get_logger
from common.metrics import MetricsCollector


class HardwareAccelerator:
    """Generic hardware accelerator interface."""

    def __init__(self, accelerator_type: str = "generic") -> None:
        self.accelerator_type = accelerator_type
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.metrics = MetricsCollector.get_instance()

    def is_available(self) -> bool:  # pragma: no cover - simple default
        """Return whether the accelerator is available."""
        return False

    def get_device_info(self) -> Dict[str, Any]:  # pragma: no cover - simple default
        """Return information about the accelerator device."""
        return {"accelerator_type": self.accelerator_type}
