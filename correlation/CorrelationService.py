import logging
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ww import WWResult, WWResultService


class CorrelationService:
    @staticmethod
    def train_random_forest_on_summary_and_accuracy(analysis_results: List[WWResult]):
        logging.log(logging.INFO, f"Training random forest regressor...")
        metrics = WWResultService.extract_summary_metrics(analysis_results)
        dataframe = pd.DataFrame.from_dict(metrics)
        split_index = len(dataframe) // 2
        train_x, test_x = dataframe[:split_index], dataframe[split_index:]
        accuracies = [result.model_accuracy for result in analysis_results]
        train_y, test_y = accuracies[:split_index], accuracies[split_index:]

        regressor = RandomForestRegressor(max_depth=2, random_state=0, max_features="auto")
        regressor.fit(train_x, train_y)
        coefficient_of_determination = regressor.score(test_x, test_y)

        logging.log(logging.INFO, "Trained random forest regressor with coefficient of determination of "
                                  f"prediction: {coefficient_of_determination}")

