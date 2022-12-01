from enum import auto, Enum
from typing import Dict

import tensorflow as tf

from models import ModelRegistry, ModelWrapper, ModelDescriptor, ModelTag


class RegNetXVariant(Enum):
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


RegNetXTop1Accuracy: Dict[RegNetXVariant, float] = {
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


def register_regnetx_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in RegNetXVariant:
        identification = "RegNet" + variant.name + ("-pretrained" if pretrained else "-untrained")
        name = "RegNet" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "RegNetX",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): RegNetXTop1Accuracy[variant]
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
