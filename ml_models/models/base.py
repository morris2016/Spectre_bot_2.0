from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Iterable

DataBatch = Tuple[Any, Any]

@dataclass
class ModelConfig:
    """Generic model configuration."""
    name: str = "base_model"
    device: str = "cpu"
    parameters: Optional[Dict[str, Any]] = None


    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ModelOutput:
    """Model prediction output container."""
    predictions: Any
    probabilities: Any = None
    raw_output: Any = None

class BaseModel:
    """Base class for machine learning models."""


    def __init__(self, config: Optional[ModelConfig] = None,
                 name: str = "base", **kwargs: Any) -> None:
        """Initialize the model with configuration and arbitrary options."""
        self.config = config or ModelConfig()
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train_batch(self, batch: DataBatch) -> Dict[str, float]:
        raise NotImplementedError

    def predict_batch(self, batch: Any) -> ModelOutput:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
