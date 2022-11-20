from enum import auto
from typing import Dict, Optional

import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2

from .ModelIdentification import ModelArchitecture
from .ModelWrapperBase import ModelWrapperBase, ModelVariant


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
        self._model_constructor = getattr(tf.keras.applications, "EfficientNetV2" + variant.name)
        self._model: Optional[EfficientNetV2] = None

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self._model_constructor(include_top=False, include_preprocessing=False, weights="imagenet")
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return EfficientNetV2Top1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.EfficientNetV2
