import logging
from typing import List
import weightwatcher as ww

from .WeightWatcherResult import WeightWatcherResult, WeightWatcherSummary
from models import ModelWrapperBase


class WeightWatcherService:
    def __init__(self, log_level=logging.WARNING):
        self._log_level = log_level
        self._weight_watcher = ww.WeightWatcher(log_level=log_level)

    def analyze_model(self, model_wrapper: ModelWrapperBase) -> WeightWatcherResult:
        logging.log(logging.INFO, f"Analyzing {model_wrapper.identification}")
        result = self._get_details_and_summary(model_wrapper)
        logging.log(logging.INFO, f"Summary {model_wrapper.identification}: {result.summary}")
        return result

    def analyze_models(self, model_wrappers: List[ModelWrapperBase]) -> List[WeightWatcherResult]:
        results = []
        for model_wrapper in model_wrappers:
            results.append(self.analyze_model(model_wrapper))
        return results

    def _get_details_and_summary(self, model_wrapper: ModelWrapperBase) -> WeightWatcherResult:
        details = self._weight_watcher.analyze(model=model_wrapper.model)
        summary = WeightWatcherSummary(**self._weight_watcher.get_summary(details))
        return WeightWatcherResult(model_wrapper.identification, model_wrapper.top_1_accuracy, summary, details)

