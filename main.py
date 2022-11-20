from weight_watcher import WeightWatcherResultService
from visualization import PlottingService, DashboardService
from typing import List
import plotly.graph_objects as go
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plotting_service = PlottingService()
dashboard_service = DashboardService()
analysis_result_repository = WeightWatcherResultService('results')


def main():
    results = analysis_result_repository.load_all()

    plots: List[go.Figure] = []
    plots += [plotting_service.create_summaries_figure(results, group_by_variant=False)]
    plots += plotting_service.create_summaries_per_architecture_figures(results)
    plots += [plotting_service.create_details_figure(results, group_by_architecture=True, group_by_variant=False)]
    plots += plotting_service.create_details_per_architecture_figures(results)
    plots += plotting_service.create_details_per_model_figures(results)

    dashboard_service.build_dashboard(plots)
    dashboard_service.show_dashboard(True)


if __name__ == '__main__':
    main()
