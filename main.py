from typing import List

import plotly.graph_objects as go

from visualization import PlottingService, DashboardService
from ww import WWResultRepository


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


def main():
    show_all_plots()


if __name__ == '__main__':
    main()
