from __future__ import annotations

import logging
from typing import Dict, List, Callable
import timm
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import ModelDescriptor, ModelWrapper
import tensorflow as tf


class ModelRegistry:
    _instance = None

    def __init__(self):
        self._model_wrappers: Dict[ModelDescriptor, ModelWrapper] = dict()

    def register_model(self, model_descriptor: ModelDescriptor, model_wrapper: ModelWrapper):
        self._model_wrappers[model_descriptor] = model_wrapper

    def register_timm_model(self, name: str, pretrained: bool):
        assert timm.is_model(name)
        model_descriptor = ModelDescriptor(
            id=name,
            tags={
                "timm_model": name,
                "pretrained": pretrained,
            }
        )
        model_wrapper = ModelWrapper(
            descriptor=model_descriptor,
            model_factory=lambda: timm.create_model(model_name=name, pretrained=pretrained)
        )

        self.register_model(model_descriptor, model_wrapper)

    def get_model(self, model_descriptor: ModelDescriptor) -> ModelWrapper:
        if model_descriptor not in self._model_wrappers:
            raise KeyError(f"{str(model_descriptor)} has no registered model factory")
        return self._model_wrappers[model_descriptor]

    def get_models_by_selector(self, model_selector: Callable[[ModelDescriptor], bool]) -> List[ModelWrapper]:
        model_descriptors = [descriptor for descriptor in self._model_wrappers if model_selector(descriptor)]
        model_wrappers = [self._model_wrappers[descriptor] for descriptor in model_descriptors]
        return model_wrappers

    def get_all_models(self) -> List[ModelWrapper]:
        return self._model_wrappers.values()

    @classmethod
    def instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    @staticmethod
    def get_timm(model_name: str):
        # All tim models are pretrained on ImageNet ??
        model = timm.create_model(model_name, pretrained=True)

    @staticmethod
    def list_pretrained_timm_models():
        return timm.list_models(pretrained=True)
