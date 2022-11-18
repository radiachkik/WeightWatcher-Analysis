from enum import auto
from typing import Dict

import tensorflow as tf
from keras.applications.efficientnet import EfficientNet

from models.ModelArchitecture import ModelArchitecture
from models.ModelWrapperBase import ModelWrapperBase, ModelVariant


class EfficientNetVariant(ModelVariant):
    B0 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    B4 = auto()
    B5 = auto()
    B6 = auto()
    B7 = auto()


EfficientNetTop1Accuracy: Dict[ModelVariant, float] = {
    EfficientNetVariant.B0: 77.2,
    EfficientNetVariant.B1: 79.1,
    EfficientNetVariant.B2: 80.2,
    EfficientNetVariant.B3: 81.6,
    EfficientNetVariant.B4: 83.0,
    EfficientNetVariant.B5: 83.7,
    EfficientNetVariant.B6: 84.1,
    EfficientNetVariant.B7: 84.4,
}


class EfficientNetWrapper(ModelWrapperBase):
    def __init__(self, variant: EfficientNetVariant):
        super().__init__(variant)
        model_module = tf.keras.applications
        constructor = getattr(model_module, "EfficientNet" + variant.name)
        self._model: EfficientNet = constructor(include_top=False, weights="imagenet")

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return EfficientNetTop1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.EfficientNet
