import logging
from typing import List

import plotly.graph_objects as go

from models import ModelQueryService, configure_gpu, configure_cpu
from hardcoded_models import register_hardcoded_models
from visualization import PlottingService, DashboardService
from ww import WWResultRepository, WWService


def show_all_plots():
    plotting_service = PlottingService()
    dashboard_service = DashboardService()
    ww_result_repository = WWResultRepository('results')

    results = ww_result_repository.get_all()

    plots: List[go.Figure] = []
    plots += [plotting_service.create_summaries_figure(results, group_by_variant=False)]
    plots += plotting_service.create_summaries_per_architecture_figures(results)
    plots += [plotting_service.create_details_figure(results, group_by_architecture=True, group_by_variant=False)]
    plots += plotting_service.create_details_per_architecture_figures(results)
    plots += plotting_service.create_details_per_model_figures(results)

    dashboard_service.build_dashboard(plots)
    dashboard_service.show_dashboard(True)


def analyze_all_models():
    logging.basicConfig(level=logging.INFO)
    # configure_gpu()
    configure_cpu()
    register_hardcoded_models(pretrained=True)
    register_hardcoded_models(pretrained=False)
    ww_service = WWService()
    ww_result_repository = WWResultRepository('ww_results')
    model_query_service = ModelQueryService()

    model_wrappers = model_query_service.get_all()
    ww_results = ww_service.analyze_models(model_wrappers, ww_result_repository)
    ww_result_repository.add_many(ww_results)


def main():
    analyze_all_models()


if __name__ == '__main__':
    main()
