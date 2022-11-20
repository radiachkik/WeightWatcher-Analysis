from typing import Dict, Type, List

from .ConvNeXt import ConvNeXtVariant, ConvNeXtWrapper
from .EfficientNet import EfficientNetVariant, EfficientNetWrapper
from .EfficientNetV2 import EfficientNetV2Variant, EfficientNetV2Wrapper
from .ModelIdentification import ModelArchitecture, ModelIdentification
from .ModelWrapperBase import ModelVariant, ModelWrapperBase
from .RegNetX import RegNetXVariant, RegNetXWrapper
from .RegNetY import RegNetYVariant, RegNetYWrapper
from .ResNetRS import ResNetRSVariant, ResNetRSWrapper


ModelArchitectureToVariantsMapping: Dict[ModelArchitecture, Type[ModelVariant]] = {
    ModelArchitecture.ConvNeXt: ConvNeXtVariant,
    ModelArchitecture.EfficientNet: EfficientNetVariant,
    ModelArchitecture.EfficientNetV2: EfficientNetV2Variant,
    ModelArchitecture.RegNetX: RegNetXVariant,
    ModelArchitecture.RegNetY: RegNetYVariant,
    ModelArchitecture.ResNetRS: ResNetRSVariant
}

ModelArchitectureToWrapperClassMapping: Dict[ModelArchitecture, Type[ModelWrapperBase]] = {
    ModelArchitecture.ConvNeXt: ConvNeXtWrapper,
    ModelArchitecture.EfficientNet: EfficientNetWrapper,
    ModelArchitecture.EfficientNetV2: EfficientNetV2Wrapper,
    ModelArchitecture.RegNetX: RegNetXWrapper,
    ModelArchitecture.RegNetY: RegNetYWrapper,
    ModelArchitecture.ResNetRS: ResNetRSWrapper
}


class ModelService:
    @staticmethod
    def get(model_identification: ModelIdentification) -> ModelWrapperBase:
        ModelService.verify_model_identification(model_identification)
        model_wrapper_class = ModelArchitectureToWrapperClassMapping[model_identification.architecture]
        model_wrapper = model_wrapper_class(model_identification.variant)
        return model_wrapper

    @staticmethod
    def get_all_of_architecture(model_architecture: ModelArchitecture) -> List[ModelWrapperBase]:
        ModelService.verify_model_architecture(model_architecture)
        wrapper_class = ModelArchitectureToWrapperClassMapping[model_architecture]
        variants_type = ModelArchitectureToVariantsMapping[model_architecture]
        model_wrappers = [wrapper_class(variant) for variant in variants_type]
        return model_wrappers

    @staticmethod
    def get_all() -> List[ModelWrapperBase]:
        model_wrappers = []
        for model_architecture in ModelArchitecture:
            wrapper_class = ModelArchitectureToWrapperClassMapping[model_architecture]
            variants_type = ModelArchitectureToVariantsMapping[model_architecture]
            model_wrappers += [wrapper_class(variant) for variant in variants_type]
        return model_wrappers

    @staticmethod
    def get_model_variant_by_name(model_architecture: ModelArchitecture, variant_name: str) -> ModelVariant:
        ModelService.verify_model_architecture(model_architecture)
        variants_type = ModelArchitectureToVariantsMapping[model_architecture]
        variant_names = [variant.name for variant in variants_type]
        if variant_name not in variant_names:
            raise ValueError(f"There is no variant '{variant_name}' for model architecture '{model_architecture.name}'")
        return variants_type[variant_name]

    @staticmethod
    def verify_model_identification(model_identification: ModelIdentification):
        try:
            accepted_model_variants_type = ModelArchitectureToVariantsMapping[model_identification.architecture]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported model type")

        try:
            accepted_model_variants_type[model_identification.variant.name]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported variant for model type '{model_identification.architecture.name}'")

    @staticmethod
    def verify_model_architecture(model_architecture: ModelArchitecture):
        try:
            ModelArchitectureToVariantsMapping[model_architecture]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported model type")


