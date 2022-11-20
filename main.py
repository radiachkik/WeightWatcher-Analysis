import logging
import os

from models import ModelService
from weight_watcher import WeightWatcherService, AnalysisResultService
from analysis import AnalysisService

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
weight_watcher_service = WeightWatcherService(logging.WARNING)
analysis_result_repository = AnalysisResultService('results')
analysis_service = AnalysisService()


def analyze_all_models():
    efficientnet_wrappers = ModelService.get_all()
    results = weight_watcher_service.analyze_models(efficientnet_wrappers)
    analysis_result_repository.save_many(results)
    analysis_service.plot_accuracy_correlation(results)


def report_all_existing_analysis_results():
    results = analysis_result_repository.load_all()
    analysis_service.plot_accuracy_correlation(results)


def train_svm():
    results = analysis_result_repository.load_all()
    analysis_service.train_svm_on_summary_correlation_with_accuracy(results)


def main():
    train_svm()


if __name__ == '__main__':
    main()
