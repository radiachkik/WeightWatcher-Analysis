from enum import auto, Enum
from typing import Dict

import tensorflow as tf

from models import ModelRegistry, ModelDescriptor, ModelWrapper, ModelTag


class EfficientNetV2Variant(Enum):
    B0 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    S = auto()
    M = auto()
    L = auto()


EfficientNetV2Top1Accuracy: Dict[EfficientNetV2Variant, float] = {
    EfficientNetV2Variant.B0: 78.7,
    EfficientNetV2Variant.B1: 79.8,
    EfficientNetV2Variant.B2: 81.8,  # ???? This is just estimated
    EfficientNetV2Variant.B3: 82.1,
    EfficientNetV2Variant.S: 83.9,
    EfficientNetV2Variant.M: 85.1,
    EfficientNetV2Variant.L: 85.7
}


def register_efficientnet_v2_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in EfficientNetV2Variant:
        identification = "EfficientNetV2" + variant.name + "-pretrained" if pretrained else "-untrained"
        name = "EfficientNetV2" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "EfficientNetV2",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): EfficientNetV2Top1Accuracy[variant]
            }
        )

        model_wrapper = ModelWrapper(
            descriptor=model_descriptor,
            model_factory=lambda: getattr(tf.keras.applications, name)(
                include_top=False,
                include_preprocessing=False,
                weights="imagenet" if pretrained else None
            )
        )

        model_registry.register_model(model_descriptor, model_wrapper)

