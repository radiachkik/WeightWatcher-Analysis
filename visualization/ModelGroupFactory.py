from typing import List, Generator

from models import ModelTag
from visualization import ModelGroup
from ww import WWResult


class ModelGroupFactory:
    @staticmethod
    def architecture_group_generator(ww_results: List[WWResult]) -> Generator[ModelGroup, None, None]:
        architectures = [result.model_descriptor.tags.get(str(ModelTag.ARCHITECTURE)) for result in ww_results]
        architectures = [architecture for architecture in architectures if architecture is not None]
        architectures= list(set(architectures))

        for architecture in architectures:
            def architecture_selector(ww_result: WWResult) -> bool:
                model_architecture = ww_result.model_descriptor.tags.get(str(ModelTag.ARCHITECTURE))
                return model_architecture == architecture

            yield ModelGroup(name=architecture, selector=architecture_selector)
