from typing import Dict, Type, List

from models.ConvNeXt import ConvNeXtVariant, ConvNeXtWrapper
from models.EfficientNet import EfficientNetVariant, EfficientNetWrapper
from models.EfficientNetV2 import EfficientNetV2Variant, EfficientNetV2Wrapper
from models.ModelArchitecture import ModelArchitecture
from models.ModelWrapperBase import ModelVariant, ModelWrapperBase
from models.RegNetX import RegNetXVariant, RegNetXWrapper
from models.RegNetY import RegNetYVariant, RegNetYWrapper
from models.ResNetRS import ResNetRSVariant, ResNetRSWrapper


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


class ModelWrapperService:
    @staticmethod
    def get_model_wrapper(model_architecture: ModelArchitecture, model_variant: ModelVariant) -> ModelWrapperBase:
        ModelWrapperService.verify_model_architecture_and_variant(model_architecture, model_variant)
        model_wrapper_class = ModelArchitectureToWrapperClassMapping[model_architecture]
        model_wrapper = model_wrapper_class(model_variant)
        return model_wrapper

    @staticmethod
    def get_all_variants_of_type(model_architecture: ModelArchitecture) -> List[ModelWrapperBase]:
        ModelWrapperService.verify_model_architecture(model_architecture)
        wrapper_class = ModelArchitectureToWrapperClassMapping[model_architecture]
        variants_type = ModelArchitectureToVariantsMapping[model_architecture]
        model_wrappers = [wrapper_class(variant) for variant in variants_type]
        return model_wrappers

    @staticmethod
    def get_all_variants_of_all_types() -> List[ModelWrapperBase]:
        model_wrappers = []
        for model_architecture in ModelArchitecture:
            wrapper_class = ModelArchitectureToWrapperClassMapping[model_architecture]
            variants_type = ModelArchitectureToVariantsMapping[model_architecture]
            model_wrappers += [wrapper_class(variant) for variant in variants_type]
        return model_wrappers

    @staticmethod
    def verify_model_architecture_and_variant(model_architecture: ModelArchitecture, model_variant: ModelVariant):
        try:
            accepted_model_variants_type = ModelArchitectureToVariantsMapping[model_architecture]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported model type")

        try:
            accepted_model_variants_type[model_variant.name]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported variant for model type '{model_architecture.name}'")

    @staticmethod
    def verify_model_architecture(model_architecture: ModelArchitecture):
        try:
            ModelArchitectureToVariantsMapping[model_architecture]
        except KeyError as e:
            raise TypeError(f"'{e.args[0]}' is not a supported model type")
