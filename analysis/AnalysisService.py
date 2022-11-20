import logging
from dataclasses import asdict
from typing import List, Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from weight_watcher import WeightWatcherResult


class AnalysisService:
    @staticmethod
    def plot_accuracy_correlation(analysis_results: List[WeightWatcherResult],
                                  figure_title: str = "Correlation with accuracy"):
        metrics = AnalysisService._extract_summary_metrics(analysis_results)
        accuracies = [result.model_accuracy for result in analysis_results]
        identifications = [result.model_identification for result in analysis_results]
        names = [f"{identification.architecture.name}/{identification.variant.name}" for identification in
                 identifications]

        fig = make_subplots(rows=2, cols=3)
        for index, key in enumerate(metrics):
            row = (index % 2) + 1
            col = (index % 3) + 1
            fig.add_trace(go.Scatter(
                x=metrics[key],
                y=accuracies,
                text=names,
                mode="markers+text",
                marker=dict(
                    size=10
                ),
            ), row=row, col=col)
            fig.update_xaxes(title_text=key, row=row, col=col)
        fig.update_yaxes(title_text="accuracy")
        fig.update_layout(title_text=figure_title)
        fig.update_traces(textposition='top center')
        fig.show()

    @staticmethod
    def train_svm_on_summary_correlation_with_accuracy(analysis_results: List[WeightWatcherResult]):
        logging.log(logging.INFO, f"Training random forest regressor...")
        metrics = AnalysisService._extract_summary_metrics(analysis_results)
        dataframe = pd.DataFrame.from_dict(metrics)
        split_index = len(dataframe) // 2
        train_x, test_x = dataframe[:split_index], dataframe[split_index:]
        accuracies = [result.model_accuracy for result in analysis_results]
        train_y, test_y = accuracies[:split_index], accuracies[split_index:]

        regressor = RandomForestRegressor(max_depth=2, random_state=0, max_features="auto")
        regressor.fit(train_x, train_y)
        coefficient_of_determination = regressor.score(test_x, test_y)

        logging.log(logging.INFO, "Trained random foresr regressor with coefficient of determination of "
                                  f"prediction: {coefficient_of_determination}")

    @staticmethod
    def _extract_summary_metrics(analysis_results: List[WeightWatcherResult]) -> Dict[str, Any]:
        metrics = dict()
        for result in analysis_results:
            result_dict = asdict(result.summary)
            for key in result_dict:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result_dict[key])
        return metrics
