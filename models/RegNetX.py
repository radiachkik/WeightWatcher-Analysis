from enum import auto
from typing import Dict, Optional

import tensorflow as tf
from keras.applications.regnet import RegNet

from . import ResNetRS
from .ModelIdentification import ModelArchitecture
from .ModelWrapperBase import ModelWrapperBase, ModelVariant


class RegNetXVariant(ModelVariant):
    X002 = auto()
    X004 = auto()
    X006 = auto()
    X008 = auto()
    X016 = auto()
    X032 = auto()
    X040 = auto()
    X064 = auto()
    X080 = auto()
    X120 = auto()
    X160 = auto()
    X320 = auto()


RegNetXTop1Accuracy: Dict[ModelVariant, float] = {
    RegNetXVariant.X002: 68.9,
    RegNetXVariant.X004: 72.6,
    RegNetXVariant.X006: 74.1,
    RegNetXVariant.X008: 75.2,
    RegNetXVariant.X016: 77.0,
    RegNetXVariant.X032: 78.3,
    RegNetXVariant.X040: 78.6,
    RegNetXVariant.X064: 79.2,
    RegNetXVariant.X080: 79.3,
    RegNetXVariant.X120: 79.7,
    RegNetXVariant.X160: 80.0,
    RegNetXVariant.X320: 80.5,
}


class RegNetXWrapper(ModelWrapperBase):
    def __init__(self, variant: RegNetXVariant):
        super().__init__(variant)
        self._model_constructor = getattr(tf.keras.applications, "RegNet" + variant.name)
        self._model: Optional[RegNet] = None

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self._model_constructor(include_top=False, include_preprocessing=False, weights="imagenet")
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return RegNetXTop1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.RegNetX
