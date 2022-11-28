import math
from typing import List, Dict

import pandas
from pandas import DataFrame
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from visualization import PlotConfiguration, FigureConfiguration
from ww import WWResult, WWDetailsColumns, WWSummaryColumns

RELEVANT_DETAILS_COLUMNS = [
    WWDetailsColumns.ALPHA.value,
    WWDetailsColumns.ALPHA_WEIGHTED.value,
    WWDetailsColumns.LOG_ALPHA_NORM.value,
    WWDetailsColumns.NORM.value,
    WWDetailsColumns.LOG_NORM.value,
    WWDetailsColumns.SPECTRAL_NORM.value,
    WWDetailsColumns.LOG_SPECTRAL_NORM.value,
    WWDetailsColumns.D.value,
    WWDetailsColumns.N.value,
    WWDetailsColumns.SIGMA.value,
    WWDetailsColumns.STABLE_RANK.value,
    WWDetailsColumns.NUM_PL_SPIKES.value,
    WWDetailsColumns.LAMBDA_MAX.value,
    WWDetailsColumns.SV_MAX.value,
    WWDetailsColumns.XMAX.value,
    WWDetailsColumns.XMIN.value,
]

RELEVANT_SUMMARY_COLUMNS = [
    WWSummaryColumns.LOG_NORM.value,
    WWSummaryColumns.ALPHA.value,
    WWSummaryColumns.ALPHA_WEIGHTED.value,
    WWSummaryColumns.LOG_ALPHA_NORM.value,
    WWSummaryColumns.LOG_SPECTRAL_NORM.value,
    WWSummaryColumns.STABLE_RANK.value,
]

COLUMNS_SUMMARY_FIGURE = 1
COLUMNS_DETAILS_FIGURE = 1

COLORS = px.colors.qualitative.Plotly


class PlottingService:
    @staticmethod
    def create_summaries_per_architecture_figures(results: List[WWResult]) -> List[go.Figure]:
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
    def create_scatter_plot(results: List[WWResult], plot_config: PlotConfiguration) -> go.Scatter:
        results = [result for result in results if plot_config.model_group.selector(result)]
        x_data = [plot_config.x_data_selector(result) for result in results]
        y_data = [plot_config.y_data_selector(result) for result in results]
        text_data = [plot_config.text_data_selector(result) for result in results] if plot_config.text else None

        modes = []
        if plot_config.markers:
            modes.append("markers")
        if plot_config.text:
            modes.append("text")
        if plot_config.lines:
            modes.append("lines")
        mode = "+".join(modes)
        scatter = go.Scatter(
            x=x_data,
            y=y_data,
            text=text_data,
            mode=mode,
            marker=dict(
                size=plot_config.marker_size,
            ) if plot_config.marker_size is not None else None,
            legendgroup=plot_config.legend_group,
            legendgrouptitle={"text": plot_config.legend_group},
            name=plot_config.plot_name,
            showlegend=True,
        )
        return scatter

    @staticmethod
    def create_figure(results: List[WWResult], fig_config: FigureConfiguration):
        fig = make_subplots(rows=fig_config.num_rows, cols=fig_config.num_cols)
        for plot_config, row, col in zip(fig_config.plot_configs, fig_config.row_indices, fig_config.col_indices):
            scatter = PlottingService.create_scatter_plot(results, plot_config)
            fig.add_trace(scatter, row=row, col=col)
            fig.update_xaxes(title_text=plot_config.x_axis_title, row=row, col=col)
            fig.update_yaxes(title_text=plot_config.y_axis_title, row=row, col=col)

        fig.update_layout(title_text=fig_config.figure_name, height=fig_config.height, width=fig_config.width)
        fig.update_traces(textposition='top center')
        return fig

    @staticmethod
    def create_summaries_figure(
            results: List[WWResult],
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
                    y=df[WWSummaryColumns.ACCURACY.value],
                    text=df[WWSummaryColumns.ARCHITECTURE.value] + ":" + df[WWSummaryColumns.VARIANT.value],
                    mode="markers+text+lines",
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
    def create_details_per_architecture_figures(results: List[WWResult]) -> List[go.Figure]:
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
    def create_details_per_model_figures(results: List[WWResult]) -> List[go.Figure]:
        figures = []
        for result in results:
            fig = PlottingService.create_details_figure(
                [result],
                lines=True,
                group_by_architecture=True,
                group_by_variant=True,
                figure_title=f"Details [{result.model_descriptor}]"
            )
            figures.append(fig)
        return figures

    @staticmethod
    def create_details_figure(
            results: List[WWResult],
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
                    x=df[WWDetailsColumns.LAYER_ID.value],
                    y=df[metric_key],
                    text=df[WWDetailsColumns.ARCHITECTURE.value] + ":" + df[
                        WWDetailsColumns.VARIANT.value],
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

            fig.update_xaxes(title_text=WWDetailsColumns.LAYER_ID.value)
            fig.update_layout(title_text=figure_title, height=rows * 400)
            fig.update_traces(textposition='top center')
        return fig

    @staticmethod
    def group_results_by_architecture(results: List[WWResult]) -> Dict[str, List[WWResult]]:
        architectures = {}
        for result in results:
            architecture_name = result.model_descriptor.architecture.id
            if architecture_name not in architectures:
                architectures[architecture_name] = []
            architectures[architecture_name].append(result)
        return architectures

    @staticmethod
    def split_dataframe(df: DataFrame, group_by_architecture: bool, group_by_variant: bool):
        names = ["all"]
        trace_dfs = [df]
        if group_by_architecture:
            architecture_groups = df.groupby(WWDetailsColumns.ARCHITECTURE.value)
            trace_dfs = [architecture_group[1] for architecture_group in architecture_groups]
            names = [architecture_group[0] for architecture_group in architecture_groups]

        if group_by_variant:
            new_trace_dfs = []
            variant_names = []
            for group_name, trace_df in zip(names, trace_dfs):
                variant_groups = trace_df.groupby(WWDetailsColumns.VARIANT.value)
                new_trace_dfs += [variant_group[1] for variant_group in variant_groups]
                variant_names += [f"{group_name}:{variant_group[0]}" if group_by_architecture else variant_group[0] for
                                  variant_group in variant_groups]
            trace_dfs = new_trace_dfs
            names = variant_names

        return names, trace_dfs

    @staticmethod
    def split_df(df: DataFrame, group_by_architecture: bool, group_by_variant: bool):
        names = ["all"]
        trace_dfs = [df]
        if group_by_architecture:
            architecture_groups = df.groupby(WWDetailsColumns.ARCHITECTURE.value)
            trace_dfs = [architecture_group[1] for architecture_group in architecture_groups]
            names = [architecture_group[0] for architecture_group in architecture_groups]

        if group_by_variant:
            new_trace_dfs = []
            variant_names = []
            for group_name, trace_df in zip(names, trace_dfs):
                variant_groups = trace_df.groupby(WWDetailsColumns.VARIANT.value)
                new_trace_dfs += [variant_group[1] for variant_group in variant_groups]
                variant_names += [f"{group_name}:{variant_group[0]}" if group_by_architecture else variant_group[0] for
                                  variant_group in variant_groups]
            trace_dfs = new_trace_dfs
            names = variant_names

        return names, trace_dfs
