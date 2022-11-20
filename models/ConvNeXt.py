from enum import auto
from typing import Dict, Optional

import tensorflow as tf
from keras.applications.convnext import ConvNeXt

from .ModelIdentification import ModelArchitecture
from .ModelWrapperBase import ModelWrapperBase, ModelVariant


class ConvNeXtVariant(ModelVariant):
    Tiny = auto()
    Small = auto()
    Base = auto()
    Large = auto()
    XLarge = auto()


ConvNeXtTop1Accuracy: Dict[ModelVariant, float] = {
    ConvNeXtVariant.Tiny: 82.1,
    ConvNeXtVariant.Small: 83.1,
    ConvNeXtVariant.Base: 83.8,
    ConvNeXtVariant.Large: 84.3,
    ConvNeXtVariant.XLarge: 85.5
}


class ConvNeXtWrapper(ModelWrapperBase):
    def __init__(self, variant: ConvNeXtVariant):
        super().__init__(variant)
        self._model_constructor = getattr(tf.keras.applications, "ConvNeXt" + variant.name)
        self._model: Optional[ConvNeXt] = None

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self._model_constructor(include_top=False, include_preprocessing=False, weights="imagenet")
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return ConvNeXtTop1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.ConvNeXt


