from enum import auto, Enum
from typing import Dict

import tensorflow as tf

from models import ModelDescriptor, ModelRegistry, ModelWrapper, ModelTag


class RegNetYVariant(Enum):
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


RegNetYTop1Accuracy: Dict[RegNetYVariant, float] = {
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


def register_regnety_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in RegNetYVariant:
        identification = "RegNet" + variant.name + "-pretrained" if pretrained else "-untrained"
        name = "RegNet" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "RegNetY",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): RegNetYTop1Accuracy[variant]
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
