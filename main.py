import logging
from typing import List

import plotly.graph_objects as go

from application import AnalyzeService
from models import ModelQueryService, configure_cpu, ModelTag
from hardcoded_models import register_hardcoded_models
from visualization import PlottingService, DashboardService, FigureConfiguration, PlotConfiguration, \
    ModelGroupFactory
from ww import WWResultRepository, WWService, WWResult, WWSummaryColumns


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

def show_plot():
    plotting_service = PlottingService()
    dashboard_service = DashboardService()
    ww_result_repository = WWResultRepository('results')
    ww_results = ww_result_repository.get_all()

    def x_data_selector(ww_result: WWResult) -> float:
        return ww_result.summary[WWSummaryColumns.LOG_NORM.value]

    def y_data_selector(ww_result: WWResult) -> float:
        return ww_result.model_descriptor.tags.get(str(ModelTag.ARCHITECTURE.value)) or 0.0

    def text_data_selector(ww_result: WWResult) -> str:
        return ww_result.model_descriptor.id

    plot_configs = []
    row_indices = []
    col_indices = []

    for index, model_group in enumerate(ModelGroupFactory.architecture_group_generator(ww_results)):
        plot_config = PlotConfiguration(
            plot_name=model_group.name,
            legend_group=model_group.name,
            result_selector=model_group.selector,
            x_data_selector=x_data_selector,
            y_data_selector=y_data_selector,
            text_data_selector=text_data_selector,
            markers=True,
            text=True,
            lines=False,
            marker_size=10,
            x_axis_title="alpha",
            y_axis_title="accuracy"
        )
        plot_configs.append(plot_config)
        row_indices.append(1)
        col_indices.append(1)

    figure_config = FigureConfiguration(
        figure_name="All pretrained models",
        num_rows=1,
        num_cols=1,
        plot_configs=plot_configs,
        row_indices=[],
        col_indices=[],
        height=500,
        width=1000
    )
    figure = plotting_service.create_figure(ww_results, figure_config)

    dashboard_service.build_dashboard([figure])
    dashboard_service.show_dashboard(True)


def main():
    show_plot()


if __name__ == '__main__':
    main()
