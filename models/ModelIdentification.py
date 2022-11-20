from dataclasses import dataclass
from enum import Enum, auto


class ModelArchitecture(Enum):
    ConvNeXt = auto()
    EfficientNet = auto()
    EfficientNetV2 = auto()
    RegNetX = auto()
    RegNetY = auto()
    ResNetRS = auto()


class ModelVariant(Enum):
    pass


@dataclass
class ModelIdentification:
    architecture: ModelArchitecture
    variant: ModelVariant

    def __str__(self):
        return f"{self.architecture.name}:{self.variant.name}"
