from abc import ABC, abstractmethod
import tensorflow as tf

from .ModelIdentification import ModelArchitecture, ModelVariant, ModelIdentification


class ModelWrapperBase(ABC):
    def __init__(self, variant: ModelVariant):
        self._variant = variant

    @property
    def variant(self) -> ModelVariant:
        return self._variant

    @property
    def identification(self) -> ModelIdentification:
        return ModelIdentification(self.architecture, self.variant)

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
