import math
from typing import List, Dict

import pandas
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from weight_watcher import WeightWatcherResult, WeightWatcherDetailsColumns, WeightWatcherSummaryColumns

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

RELEVANT_SUMMARY_COLUMNS = [
    WeightWatcherSummaryColumns.LOG_NORM.value,
    WeightWatcherSummaryColumns.ALPHA.value,
    WeightWatcherSummaryColumns.ALPHA_WEIGHTED.value,
    WeightWatcherSummaryColumns.LOG_ALPHA_NORM.value,
    WeightWatcherSummaryColumns.LOG_SPECTRAL_NORM.value,
    WeightWatcherSummaryColumns.STABLE_RANK.value,
]

COLUMNS_SUMMARY_FIGURE = 1
COLUMNS_DETAILS_FIGURE = 1

COLORS = px.colors.qualitative.Plotly


class PlottingService:
    @staticmethod
    def create_summaries_per_architecture_figures(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for architecture_name, architecture_results in PlottingService.group_results_by_architecture(results).items():
            fig = PlottingService.create_summaries_figure(
                architecture_results,
                group_by_architecture=False,
                group_by_variant=True,
                figure_title=f"Summary [{architecture_name}]"
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_summaries_figure(
            results: List[WeightWatcherResult],
            group_by_architecture=True,
            group_by_variant=True,
            figure_title: str = "Summary [all]"
    ) -> go.Figure:
        df = pandas.concat([result.summary for result in results])
        names, trace_dfs = PlottingService.split_dataframe(df, group_by_architecture, group_by_variant)
        rows = math.ceil(len(RELEVANT_SUMMARY_COLUMNS) / float(COLUMNS_SUMMARY_FIGURE))
        fig = make_subplots(rows=rows, cols=COLUMNS_SUMMARY_FIGURE)
        for column_index, column_key in enumerate(RELEVANT_SUMMARY_COLUMNS):
            col = (column_index % COLUMNS_SUMMARY_FIGURE) + 1
            row = (column_index // COLUMNS_SUMMARY_FIGURE) + 1
            for name, df in zip(names, trace_dfs):
                scatter = go.Scatter(
                    x=df[column_key],
                    y=df[WeightWatcherSummaryColumns.ACCURACY.value],
                    text=df[WeightWatcherSummaryColumns.ARCHITECTURE.value] + ":" + df[WeightWatcherSummaryColumns.VARIANT.value],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                    ),
                    legendgroup=column_key,
                    legendgrouptitle={"text": column_key},
                    name=name,
                    showlegend=True,
                )
                fig.add_trace(scatter, row=row, col=col)
                fig.update_xaxes(title_text=column_key, row=row, col=col)
        fig.update_yaxes(title_text="accuracy")
        fig.update_layout(title_text=figure_title, height=rows * 750)
        fig.update_traces(textposition='top center')
        return fig

    @staticmethod
    def create_details_per_architecture_figures(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for architecture_name, architecture_results in PlottingService.group_results_by_architecture(results).items():
            fig = PlottingService.create_details_figure(
                architecture_results,
                lines=True,
                group_by_architecture=False,
                group_by_variant=True,
                figure_title=f"Details [{architecture_name}]"
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_details_per_model_figures(results: List[WeightWatcherResult]) -> List[go.Figure]:
        figures = []
        for result in results:
            fig = PlottingService.create_details_figure(
                [result],
                lines=True,
                group_by_architecture=True,
                group_by_variant=True,
                figure_title=f"Details [{result.model_identification}]"
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_details_figure(
            results: List[WeightWatcherResult],
            lines: bool = False,
            group_by_architecture=True,
            group_by_variant=True,
            figure_title: str = "Details [all]"
    ) -> go.Figure:
        df: DataFrame = pandas.concat([result.details for result in results])
        names, trace_dfs = PlottingService.split_dataframe(df, group_by_architecture, group_by_variant)
        rows = math.ceil(len(RELEVANT_DETAILS_COLUMNS) / float(COLUMNS_DETAILS_FIGURE))
        fig = make_subplots(rows=rows, cols=COLUMNS_DETAILS_FIGURE)
        for metric_index, metric_key in enumerate(RELEVANT_DETAILS_COLUMNS):
            col = (metric_index % COLUMNS_DETAILS_FIGURE) + 1
            row = (metric_index // COLUMNS_DETAILS_FIGURE) + 1

            for name, df in zip(names, trace_dfs):
                scatter = go.Scatter(
                    x=df[WeightWatcherDetailsColumns.LAYER_ID.value],
                    y=df[metric_key],
                    text=df[WeightWatcherDetailsColumns.ARCHITECTURE.value] + ":" + df[
                        WeightWatcherDetailsColumns.VARIANT.value],
                    mode="markers+lines" if lines else "markers",
                    marker=dict(
                        size=4,
                    ),
                    legendgroup=metric_key,
                    legendgrouptitle={"text": metric_key},
                    name=name,
                    showlegend=True,
                )
                fig.add_trace(scatter, row=row, col=col)
                fig.update_yaxes(title_text=metric_key, row=row, col=col)

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

    @staticmethod
    def split_dataframe(df: DataFrame, group_by_architecture: bool, group_by_variant: bool):
        names = ["all"]
        trace_dfs = [df]
        if group_by_architecture:
            architecture_groups = df.groupby(WeightWatcherDetailsColumns.ARCHITECTURE.value)
            trace_dfs = [architecture_group[1] for architecture_group in architecture_groups]
            names = [architecture_group[0] for architecture_group in architecture_groups]

        if group_by_variant:
            new_trace_dfs = []
            variant_names = []
            for group_name, trace_df in zip(names, trace_dfs):
                variant_groups = trace_df.groupby(WeightWatcherDetailsColumns.VARIANT.value)
                new_trace_dfs += [variant_group[1] for variant_group in variant_groups]
                variant_names += [f"{group_name}:{variant_group[0]}" if group_by_architecture else variant_group[0] for
                                  variant_group in variant_groups]
            trace_dfs = new_trace_dfs
            names = variant_names

        return names, trace_dfs
