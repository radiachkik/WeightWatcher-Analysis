from enum import auto, Enum
import tensorflow as tf
from typing import Dict

from models import ModelRegistry, ModelDescriptor, ModelWrapper, ModelTag


class ResNetRSVariant(Enum):
    RS50 = auto()
    RS101 = auto()
    RS152 = auto()
    RS200 = auto()
    RS270 = auto()
    RS350 = auto()
    RS420 = auto()


ResNetRSTop1Accuracy: Dict[ResNetRSVariant, float] = {
    ResNetRSVariant.RS50: 78.8,
    ResNetRSVariant.RS101: 81.2,  # corresponds to image resolution 192 (80.3 with 160)
    ResNetRSVariant.RS152: 83.0,  # corresponds to image resolution 256 (82.8 with 224)
    ResNetRSVariant.RS200: 83.4,
    ResNetRSVariant.RS270: 83.8,
    ResNetRSVariant.RS350: 84.0,
    ResNetRSVariant.RS420: 84.4,
}


def register_resnetrs_models(pretrained: bool):
    model_registry = ModelRegistry.instance()
    for variant in ResNetRSVariant:
        identification = "ResNet" + variant.name + ("-pretrained" if pretrained else "-untrained")
        name = "ResNet" + variant.name
        model_descriptor = ModelDescriptor(
            id=identification,
            tags={
                str(ModelTag.ARCHITECTURE): "ResNetRS",
                str(ModelTag.VARIANT): variant.name,
                str(ModelTag.PRETRAINED): pretrained,
                str(ModelTag.ACCURACY): ResNetRSTop1Accuracy[variant]
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
