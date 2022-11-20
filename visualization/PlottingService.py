import math
from typing import List, Dict

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from weight_watcher import WeightWatcherResult, WeightWatcherResultService, WeightWatcherDetailsColumns

RELEVANT_DETAILS_COLUMNS = [
    WeightWatcherDetailsColumns.ALPHA.value,
    WeightWatcherDetailsColumns.ALPHA_WEIGHTED.value,
    WeightWatcherDetailsColumns.LOG_ALPHA_NORM.value,
    WeightWatcherDetailsColumns.NORM.value,
    WeightWatcherDetailsColumns.LOG_NORM.value,
    WeightWatcherDetailsColumns.SPECTRAL_NORM.value,
    WeightWatcherDetailsColumns.LOG_SPECTRAL_NORM.value,
    WeightWatcherDetailsColumns.D.value,
    WeightWatcherDetailsColumns.N.value,
    WeightWatcherDetailsColumns.SIGMA.value,
    WeightWatcherDetailsColumns.STABLE_RANK.value,
    WeightWatcherDetailsColumns.NUM_PL_SPIKES.value,
    WeightWatcherDetailsColumns.LAMBDA_MAX.value,
    WeightWatcherDetailsColumns.SV_MAX.value,
    WeightWatcherDetailsColumns.XMAX.value,
    WeightWatcherDetailsColumns.XMIN.value,
]

COLUMNS_SUMMARY_FIGURE = 1
COLUMNS_DETAILS_FIGURE = 1

COLORS = px.colors.qualitative.Plotly


class PlottingService:
    @staticmethod
    def create_summaries_per_architecture_plot(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for architecture_name, architecture_results in PlottingService.group_results_by_architecture(results).items():
            fig = PlottingService.create_summaries_plot(
                architecture_results, f"Summary [{architecture_name}]"
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_summaries_plot(results: List[WeightWatcherResult], figure_title: str = "Summary [all]") -> go.Figure:
        metrics = WeightWatcherResultService.extract_summary_metrics(results)
        accuracies = [result.model_accuracy for result in results]
        identifications = [result.model_identification for result in results]
        architectures = [identification.architecture for identification in identifications]
        rows = math.ceil(len(metrics) / float(COLUMNS_SUMMARY_FIGURE))
        fig = make_subplots(rows=rows, cols=COLUMNS_SUMMARY_FIGURE)
        for index, key in enumerate(metrics):
            col = (index % COLUMNS_SUMMARY_FIGURE) + 1
            row = (index // COLUMNS_SUMMARY_FIGURE) + 1
            fig.add_trace(go.Scatter(
                x=metrics[key],
                y=accuracies,
                text=identifications,
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=[architecture.value for architecture in architectures],
                ),
                showlegend=False,
            ), row=row, col=col)
            fig.update_xaxes(title_text=key, row=row, col=col)
        fig.update_yaxes(title_text="accuracy")
        fig.update_layout(title_text=figure_title, height=rows * 750)
        fig.update_traces(textposition='top center')
        return fig

    @staticmethod
    def create_details_per_architecture(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for architecture_name, architecture_results in PlottingService.group_results_by_architecture(results).items():
            fig = PlottingService.create_details_plots(
                architecture_results, f"Details [{architecture_name}]", True
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_details_per_model(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for result in results:
            fig = PlottingService.create_details_plots(
                [result], f"Details [{result.model_identification}]", True
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_details_plots(results: List[WeightWatcherResult], figure_title: str = "Details [all]", lines: bool = False) -> go.Figure:
        rows = math.ceil(len(RELEVANT_DETAILS_COLUMNS) / float(COLUMNS_DETAILS_FIGURE))
        fig = make_subplots(rows=rows, cols=COLUMNS_DETAILS_FIGURE)
        for column_index, column in enumerate(RELEVANT_DETAILS_COLUMNS):
            col = (column_index % COLUMNS_DETAILS_FIGURE) + 1
            row = (column_index // COLUMNS_DETAILS_FIGURE) + 1
            for result_index, result in enumerate(results):
                metric = result.details.loc[:, column]
                layer_ids = result.details.loc[:, WeightWatcherDetailsColumns.LAYER_ID.value]
                layer_names = [f"{result.model_identification}:{layer_id}" for layer_id in layer_ids]
                fig.add_trace(go.Scatter(
                    x=layer_ids,
                    y=metric,
                    text=layer_names,
                    mode="markers+lines" if lines else "markers",
                    marker=dict(
                        size=4,
                        color=COLORS[result_index % len(COLORS)],
                    ),
                    line=dict(
                        color=COLORS[result_index % len(COLORS)],
                        width=1,
                    ),
                    showlegend=False
                ), row=row, col=col)
                fig.update_yaxes(title_text=column, row=row, col=col)
            fig.update_xaxes(title_text=WeightWatcherDetailsColumns.LAYER_ID.value)
            fig.update_layout(title_text=figure_title, height=rows * 400)
            fig.update_traces(textposition='top center')
        return fig

    @staticmethod
    def group_results_by_architecture(results: List[WeightWatcherResult]) -> Dict[str, List[WeightWatcherResult]]:
        architectures = {}
        for result in results:
            architecture_name = result.model_identification.architecture.name
            if architecture_name not in architectures:
                architectures[architecture_name] = []
            architectures[architecture_name].append(result)
        return architectures
