from dataclasses import dataclass
from typing import Union

from pandas import DataFrame, Series

from models import ModelIdentification

WeightWatcherDetails = Union[DataFrame, None, Series]


@dataclass
class WeightWatcherSummary:
    log_norm: float
    alpha: float
    alpha_weighted: float
    log_alpha_norm: float
    log_spectral_norm: float
    stable_rank: float


@dataclass
class WeightWatcherResult:
    model_identification: ModelIdentification
    model_accuracy: float
    summary: WeightWatcherSummary
    details: WeightWatcherDetails
