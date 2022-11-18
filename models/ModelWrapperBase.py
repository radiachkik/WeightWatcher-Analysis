from enum import Enum
from abc import ABC, abstractmethod
import tensorflow as tf

from models.ModelArchitecture import ModelArchitecture


class ModelVariant(Enum):
    pass


class ModelWrapperBase(ABC):
    def __init__(self, variant: ModelVariant):
        self._variant = variant

    @property
    def variant(self) -> ModelVariant:
        return self._variant

    @property
    @abstractmethod
    def model(self) -> tf.keras.Model:
        raise NotImplementedError()

    @property
    @abstractmethod
    def architecture(self) -> ModelArchitecture:
        raise NotImplementedError

    @property
    @abstractmethod
    def top_1_accuracy(self) -> float:
        raise NotImplementedError()
