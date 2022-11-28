from __future__ import annotations

from typing import Callable, Union

import tensorflow as tf
import torch.nn

from . import ModelDescriptor


ModelFactory = Callable[[], Union[tf.keras.Model, torch.nn.Module]]


class ModelWrapper:
    def __init__(self, descriptor: ModelDescriptor, model_factory: ModelFactory):
        self._descriptor = descriptor
        self._model_factory = model_factory
        self._model = None

    @property
    def model(self) -> tf.keras.Model | torch.nn.Module:
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor
