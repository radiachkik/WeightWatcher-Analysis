import logging
import os

from dashboard import DashboardService
from models import ModelService
from weight_watcher import WeightWatcherService, WeightWatcherResultService
from analysis import AnalysisService

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
weight_watcher_service = WeightWatcherService(logging.WARNING)
analysis_result_repository = WeightWatcherResultService('results')
analysis_service = AnalysisService()
dashboard_service = DashboardService()


def analyze_all_models():
    efficientnet_wrappers = ModelService.get_all()
    results = weight_watcher_service.analyze_models(efficientnet_wrappers)
    analysis_result_repository.save_many(results)
    analysis_service.plot_accuracy_correlation(results)


def plot_all_existing_summaries():
    results = analysis_result_repository.load_all()
    analysis_service.plot_accuracy_correlation(results)


def plot_all_existing_summaries_by_architecture():
    results = analysis_result_repository.load_all()
    architectures = {}
    for result in results:
        architecture_name = result.model_identification.architecture.name
        if architecture_name not in architectures:
            architectures[architecture_name] = []
        architectures[architecture_name].append(result)

    for architecture_name, architecture_results in architectures.items():
        analysis_service.plot_accuracy_correlation(architecture_results, f"{architecture_name} correlation with accuracy")


def train_svm():
    results = analysis_result_repository.load_all()
    analysis_service.train_svm_on_summary_correlation_with_accuracy(results)


def show_dashboard():
    dashboard_service.add_plot("", "", "")


def main():
    show_dashboard()


if __name__ == '__main__':
    main()
