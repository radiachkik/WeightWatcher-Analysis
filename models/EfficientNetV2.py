from enum import auto
from typing import Dict

import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2

from models.ModelArchitecture import ModelArchitecture
from models.ModelWrapperBase import ModelWrapperBase, ModelVariant


class EfficientNetV2Variant(ModelVariant):
    B0 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    S = auto()
    M = auto()
    L = auto()


EfficientNetV2Top1Accuracy: Dict[ModelVariant, float] = {
    EfficientNetV2Variant.B0: 78.7,
    EfficientNetV2Variant.B1: 79.8,
    EfficientNetV2Variant.B2: 81.8,  # ???? This is just estimated
    EfficientNetV2Variant.B3: 82.1,
    EfficientNetV2Variant.S: 83.9,
    EfficientNetV2Variant.M: 85.1,
    EfficientNetV2Variant.L: 85.7
}


class EfficientNetV2Wrapper(ModelWrapperBase):
    def __init__(self, variant: EfficientNetV2Variant):
        super().__init__(variant)
        model_module = tf.keras.applications
        constructor = getattr(model_module, "EfficientNetV2" + variant.name)
        self._model: EfficientNetV2 = constructor(include_top=False, weights="imagenet", include_preprocessing=False)

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return EfficientNetV2Top1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.EfficientNetV2
