import logging
from typing import List, Optional

import pandas
import tensorflow as tf
import weightwatcher as ww

from . import WWResult, WWResultRepository
from models import ModelWrapper


class WWService:
    def __init__(self, log_level=logging.WARNING):
        self._log_level = log_level
        self._weight_watcher = ww.WeightWatcher(log_level=logging.DEBUG)

    def analyze_model(self, model_wrapper: ModelWrapper, result_repository: Optional[WWResultRepository] = None) -> Optional[WWResult]:
        logging.log(logging.INFO, f"Analyzing {model_wrapper.descriptor.id}...")
        try:
            result = self._get_details_and_summary(model_wrapper)
            if result_repository:
                result_repository.add(result)
        except tf.errors.ResourceExhaustedError:
            logging.info(f"Could not load model {model_wrapper.descriptor.id}")
            return
        logging.log(logging.INFO, f"Finished analyzing {model_wrapper.descriptor.id}")
        return result

    def analyze_models(self, model_wrappers: List[ModelWrapper], result_repository: Optional[WWResultRepository] = None) -> List[WWResult]:
        results = []
        for model_wrapper in model_wrappers:
            result = self.analyze_model(model_wrapper, result_repository)
            if result is not None:
                results.append(result)
        return results

    def get_summary_from_details(self, details):
        return pandas.DataFrame(self._weight_watcher.get_summary(details), index=[0])

    def _get_details_and_summary(self, model_wrapper: ModelWrapper) -> WWResult:
        self._weight_watcher.a
        details = self._weight_watcher.analyze(model=model_wrapper.model)
        summary = pandas.DataFrame(self._weight_watcher.get_summary(details), index=[0])
        return WWResult(model_wrapper.descriptor, summary, details)
