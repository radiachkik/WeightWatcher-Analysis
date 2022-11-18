from enum import auto
from typing import Dict

import tensorflow as tf
from keras.applications.regnet import RegNet

from models.ModelArchitecture import ModelArchitecture
from models.ModelWrapperBase import ModelWrapperBase, ModelVariant


class RegNetYVariant(ModelVariant):
    Y002 = auto()
    Y004 = auto()
    Y006 = auto()
    Y008 = auto()
    Y016 = auto()
    Y032 = auto()
    Y040 = auto()
    Y064 = auto()
    Y080 = auto()
    Y120 = auto()
    Y160 = auto()
    Y320 = auto()


RegNetYTop1Accuracy: Dict[ModelVariant, float] = {
    RegNetYVariant.Y002: 72.3,
    RegNetYVariant.Y004: 74.1,
    RegNetYVariant.Y006: 75.5,
    RegNetYVariant.Y008: 76.3,
    RegNetYVariant.Y016: 77.9,
    RegNetYVariant.Y032: 78.9,
    RegNetYVariant.Y040: 79.4,
    RegNetYVariant.Y064: 79.9,
    RegNetYVariant.Y080: 79.9,
    RegNetYVariant.Y120: 80.3,
    RegNetYVariant.Y160: 80.4,
    RegNetYVariant.Y320: 80.9,
}


class RegNetYWrapper(ModelWrapperBase):
    def __init__(self, variant: RegNetYVariant):
        super().__init__(variant)
        model_module = tf.keras.applications
        constructor = getattr(model_module, "RegNet" + variant.name)
        self._model: RegNet = constructor(include_top=False, include_preprocessing=False, weights="imagenet")

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return RegNetYTop1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.RegNetY
