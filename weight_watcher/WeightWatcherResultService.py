import json
import os.path
from dataclasses import asdict
from typing import List, Dict, Any

import pandas

from .WeightWatcherResult import WeightWatcherResult, WeightWatcherSummary
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
            with open(os.path.join(base_path, "summary.json")) as summary_file:
                summary = WeightWatcherSummary(**json.load(summary_file))
        except FileNotFoundError:
            raise ValueError()
        model_wrapper = ModelService.get(model_identification)
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

    @staticmethod
    def extract_summary_metrics(analysis_results: List[WeightWatcherResult]) -> Dict[str, Any]:
        metrics = dict()
        for result in analysis_results:
            result_dict = asdict(result.summary)
            for key in result_dict:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result_dict[key])
        return metrics

    @staticmethod
    def extract_details_metrics(analysis_results: List[WeightWatcherResult]) -> Dict[str, Any]:
        metrics = dict()
        for result in analysis_results:
            result_dict = asdict(result.summary)
            for key in result_dict:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result_dict[key])
        return metrics
