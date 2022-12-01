from typing import List

from models import ModelWrapper
from ww import WWResultRepository, WWService, WWResult


class AnalyzeService:
    @staticmethod
    def analyze(model_wrappers: List[ModelWrapper], ww_result_repository: WWResultRepository) -> List[WWResult]:
        ww_service = WWService()
        results = ww_service.analyze_models(model_wrappers, ww_result_repository)
        return results

    @staticmethod
    def resume_analyzing(model_wrappers: List[ModelWrapper], ww_result_repository: WWResultRepository) -> List[WWResult]:
        analyzed_descriptors = [result.model_descriptor for result in ww_result_repository.get_all()]
        filtered_model_wrappers = [wrapper for wrapper in model_wrappers if
                                   wrapper.descriptor not in analyzed_descriptors]
        return AnalyzeService.analyze(filtered_model_wrappers, ww_result_repository)