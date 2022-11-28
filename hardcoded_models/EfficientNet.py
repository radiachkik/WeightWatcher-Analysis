from enum import auto, Enum
from typing import Dict

import tensorflow as tf

from models import ModelWrapper, ModelDescriptor, ModelRegistry, ModelTag


class EfficientNetVariant(Enum):
    B0 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    B4 = auto()
    B5 = auto()
    B6 = auto()
    B7 = auto()


EfficientNetTop1Accuracy: Dict[EfficientNetVariant, float] = {
    EfficientNetVariant.B0: 77.2,
    EfficientNetVariant.B1: 79.1,
    EfficientNetVariant.B2: 80.2,
    EfficientNetVariant.B3: 81.6,
    EfficientNetVariant.B4: 83.0,
    EfficientNetVariant.B5: 83.7,
    EfficientNetVariant.B6: 84.1,
    EfficientNetVariant.B7: 84.4,
}


def register_efficientnet_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in EfficientNetVariant:
        identification = "EfficientNet" + variant.name + "-pretrained" if pretrained else "-untrained"
        name = "EfficientNet" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "EfficientNet",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): EfficientNetTop1Accuracy[variant]
            }
        )

        model_wrapper = ModelWrapper(
            descriptor=model_descriptor,
            model_factory=lambda: getattr(tf.keras.applications, name)(
                include_top=False,
                weights="imagenet" if pretrained else None
            )
        )

        model_registry.register_model(model_descriptor, model_wrapper)
