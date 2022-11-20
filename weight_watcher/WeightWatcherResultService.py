import json
import os.path
from dataclasses import asdict
from typing import List, Dict, Any

import pandas
from pandas import DataFrame

from .WeightWatcherResult import WeightWatcherResult, WeightWatcherSummaryColumns, WeightWatcherDetailsColumns
from models import ModelIdentification, ModelService, ModelArchitecture


class WeightWatcherResultService:
    def __init__(self, results_base_path: str):
        self._results_base_path = results_base_path

    def save(self, analysis_result: WeightWatcherResult):
        base_path = self._get_results_base_path(analysis_result.model_identification)
        os.makedirs(base_path, exist_ok=True)
        analysis_result.details.to_csv(os.path.join(base_path, "details.csv"))
        with open(os.path.join(base_path, "summary.json"), "w") as summary_file:
            json.dump(analysis_result.summary, summary_file, indent=4)

    def save_many(self, analysis_results: List[WeightWatcherResult]):
        for analysis_result in analysis_results:
            self.save(analysis_result)

    def load(self, model_identification: ModelIdentification) -> WeightWatcherResult:
        base_path = self._get_results_base_path(model_identification)
        try:
            details = pandas.read_csv(os.path.join(base_path, "details.csv"))
            summary = pandas.read_json(os.path.join(base_path, "summary.json"), orient="index").transpose()
        except FileNotFoundError:
            raise ValueError()
        model_wrapper = ModelService.get(model_identification)
        summary[WeightWatcherSummaryColumns.ACCURACY.value] = model_wrapper.top_1_accuracy
        summary[WeightWatcherSummaryColumns.ARCHITECTURE.value] = model_identification.architecture.name
        summary[WeightWatcherSummaryColumns.VARIANT.value] = model_identification.variant.name

        details[WeightWatcherDetailsColumns.ACCURACY.value] = model_wrapper.top_1_accuracy
        details[WeightWatcherDetailsColumns.ARCHITECTURE.value] = model_identification.architecture.name
        details[WeightWatcherDetailsColumns.VARIANT.value] = model_identification.variant.name
        return WeightWatcherResult(model_identification, model_wrapper.top_1_accuracy, summary, details)

    def load_all(self) -> List[WeightWatcherResult]:
        results = []
        for architecture_name in os.listdir(self._results_base_path):
            architecture_path = os.path.join(self._results_base_path, architecture_name)
            if not os.path.isdir(architecture_path):
                continue

            try:
                architecture = ModelArchitecture[architecture_name]
            except KeyError:
                continue

            for variant_name in os.listdir(architecture_path):
                variant = ModelService.get_model_variant_by_name(architecture, variant_name)
                model_identification = ModelIdentification(architecture, variant)
                results.append(self.load(model_identification))
        return results

    def _get_results_base_path(self, model_identification: ModelIdentification):
        base_path = os.path.join(
            self._results_base_path,
            model_identification.architecture.name,
            model_identification.variant.name
        )
        return base_path
