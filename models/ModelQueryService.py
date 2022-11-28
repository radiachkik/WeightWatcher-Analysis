from typing import List

from models import ModelRegistry, ModelWrapper, ModelDescriptor, ModelTag


class ModelQueryService:
    @staticmethod
    def get_all_of_architecture(model_architecture: str) -> List[ModelWrapper]:
        def model_selector(descriptor: ModelDescriptor):
            if str(ModelTag.ARCHITECTURE) not in descriptor.tags:
                return False
            return descriptor.tags[str(ModelTag.ARCHITECTURE)] == model_architecture

        model_wrappers = ModelRegistry.instance().get_models_by_selector(model_selector)
        return model_wrappers

    @staticmethod
    def get_all_untrained() -> List[ModelWrapper]:
        def model_selector(descriptor: ModelDescriptor):
            if str(ModelTag.PRETRAINED) not in descriptor:
                return False
            return descriptor.tags[str(ModelTag.PRETRAINED)] is True

        model_wrappers = ModelRegistry.instance().get_models_by_selector(model_selector)
        return model_wrappers

    @staticmethod
    def get_all() -> List[ModelWrapper]:
        model_wrappers = ModelRegistry.instance().get_all_models()
        return model_wrappers
