import math
from dataclasses import dataclass


@dataclass
class VisualizeResultsOptions:
    results_directory: str


def main(options: VisualizeResultsOptions):
    from models import ModelTag
    from visualization import PlottingService, DashboardService, FigureConfiguration, PlotConfiguration, ModelGroupFactory
    from ww import WWResultRepository, WWResult, WWSummaryColumns

    plotting_service = PlottingService()
    dashboard_service = DashboardService()
    ww_result_repository = WWResultRepository(options.results_directory)
    ww_results = ww_result_repository.get_all()

    def x_data_selector(ww_result: WWResult) -> float:
        return ww_result.summary[WWSummaryColumns.ALPHA.value].values[0]

    def y_data_selector(ww_result: WWResult) -> float:
        return ww_result.model_descriptor.tags.get(str(ModelTag.ACCURACY)) or 0.0

    def text_data_selector(ww_result: WWResult) -> str:
        return ww_result.model_descriptor.id

    plot_configs = []

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

    cols = 3
    rows = math.ceil(len(plot_configs) // 2)
    figure_config = FigureConfiguration(
        figure_name="All pretrained models",
        num_rows=rows,
        num_cols=cols,
        plot_configs=plot_configs,
        width=750 * cols
    )
    figure = plotting_service.create_figure(ww_results, figure_config)

    dashboard_service.build_dashboard([figure])
    dashboard_service.show_dashboard(True)


def get_summaries_table(ww_results):
    import pandas as pd

    summaries = []
    for result in ww_results:
        summary = result.summary.copy()
        summary["id"] = [result.model_descriptor.id]
        summaries.append(summary)
    summary_table = pd.concat(summaries, ignore_index=True)


def parse_arguments() -> VisualizeResultsOptions:
    import argparse

    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument(
        "-i",
        "--input",
        default="ww_results",
        help="Input folder to load the results from"
    )
    args = parser.parse_args()
    options = VisualizeResultsOptions(
        results_directory=args.input
    )
    return options


if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
