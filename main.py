import logging
import os

from visualization import PlottingService
from models import ModelService
from weight_watcher import WeightWatcherService, WeightWatcherResultService
from correlation import CorrelationService

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
weight_watcher_service = WeightWatcherService(logging.WARNING)
analysis_result_repository = WeightWatcherResultService('results')
correlation_service = CorrelationService()
plotting_service = PlottingService()


def analyze_all_models():
    efficientnet_wrappers = ModelService.get_all()
    results = weight_watcher_service.analyze_models(efficientnet_wrappers)
    analysis_result_repository.save_many(results)
    plotting_service.create_summaries_plot(results)


def plot_all_existing_summaries():
    results = analysis_result_repository.load_all()
    plotting_service.create_summaries_plot(results)


def plot_all_existing_summaries_by_architecture():
    results = analysis_result_repository.load_all()
    plotting_service.create_summaries_per_architecture_plot(results)


def plot_details():
    results = analysis_result_repository.load_all()
    plotting_service.create_details_per_architecture(results)


def train_svm():
    results = analysis_result_repository.load_all()
    correlation_service.train_random_forest_on_summary_and_accuracy(results)


def main():
    plot_details()


if __name__ == '__main__':
    main()
