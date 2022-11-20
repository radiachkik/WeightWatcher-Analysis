from enum import auto
import tensorflow as tf
from keras.applications.resnet_rs import ResNetRS
from typing import Dict, Optional

from .ModelIdentification import ModelArchitecture
from .ModelWrapperBase import ModelWrapperBase, ModelVariant


class ResNetRSVariant(ModelVariant):
    RS50 = auto()
    RS101 = auto()
    RS152 = auto()
    RS200 = auto()
    RS270 = auto()
    RS350 = auto()
    RS420 = auto()


ResNetRSTop1Accuracy: Dict[ModelVariant, float] = {
    ResNetRSVariant.RS50: 78.8,
    ResNetRSVariant.RS101: 81.2,  # 82.8 corresponds to image resolution 192 (80.3 with 160)
    ResNetRSVariant.RS152: 83.0,  # 82.8 corresponds to image resolution 256 (82.8 with 224)
    ResNetRSVariant.RS200: 83.4,
    ResNetRSVariant.RS270: 83.8,
    ResNetRSVariant.RS350: 84.0,
    ResNetRSVariant.RS420: 84.4,
}


class ResNetRSWrapper(ModelWrapperBase):
    def __init__(self, variant: ResNetRSVariant):
        super().__init__(variant)
        self._model_constructor = getattr(tf.keras.applications, "ResNet" + variant.name)
        self._model: Optional[ResNetRS] = None

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = self._model_constructor(include_top=False, include_preprocessing=False, weights="imagenet")
        return self._model

    @property
    def top_1_accuracy(self) -> float:
        return ResNetRSTop1Accuracy[self._variant]

    @property
    def architecture(self) -> ModelArchitecture:
        return ModelArchitecture.ResNetRS
