import json
import os.path
from dataclasses import asdict
from typing import List

import pandas

from . import WWResult
from models import ModelDescriptor


class WWResultRepository:
    def __init__(self, results_base_path: str):
        self._results_base_path = results_base_path

    def add(self, ww_result: WWResult):
        base_path = self._get_results_base_path(ww_result.model_descriptor.id)
        os.makedirs(base_path, exist_ok=True)
        ww_result.details.to_csv(os.path.join(base_path, "details.csv"))
        ww_result.summary.to_csv(os.path.join(base_path, "summary.csv"))
        with open(os.path.join(base_path, "descriptor.json"), "w") as descriptor_file:
            json.dump(asdict(ww_result.model_descriptor), descriptor_file, indent=4)

    def add_many(self, ww_results: List[WWResult]):
        for analysis_result in ww_results:
            self.add(analysis_result)

    def get(self, model_id: str) -> WWResult:
        base_path = self._get_results_base_path(model_id)
        try:
            details = pandas.read_csv(os.path.join(base_path, "details.csv"))
            summary = pandas.read_json(os.path.join(base_path, "summary.json"), orient="index").transpose()
            descriptor = ModelDescriptor(**pandas.read_json(
                os.path.join(base_path, "descriptor.json"),
                orient="index"
            ).transpose())
        except FileNotFoundError:
            raise ValueError()
        return WWResult(descriptor, summary, details)

    def get_all(self) -> List[WWResult]:
        results = []
        for model_id in os.listdir(self._results_base_path):
            model_path = os.path.join(self._results_base_path, model_id)
            if not os.path.isdir(model_path):
                continue
            try:
                results.append(self.get(model_id))
            except ValueError:
                continue
        return results

    def _get_results_base_path(self, model_id: str):
        base_path = os.path.join(
            self._results_base_path,
            model_id
        )
        return base_path
