from enum import auto, Enum
from typing import Dict

import tensorflow as tf

from models import ModelRegistry, ModelDescriptor, ModelWrapper, ModelTag


class ConvNeXtVariant(Enum):
    Tiny = auto()
    Small = auto()
    Base = auto()
    Large = auto()
    XLarge = auto()


ConvNeXtTop1Accuracy: Dict[ConvNeXtVariant, float] = {
    ConvNeXtVariant.Tiny: 82.1,
    ConvNeXtVariant.Small: 83.1,
    ConvNeXtVariant.Base: 83.8,
    ConvNeXtVariant.Large: 84.3,
    ConvNeXtVariant.XLarge: 85.5
}


def register_convnext_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in ConvNeXtVariant:
        identification = "ConvNeXt" + variant.name + ("-pretrained" if pretrained else "-untrained")
        name = "ConvNeXt" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "ConvNeXt",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): ConvNeXtTop1Accuracy[variant]
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
