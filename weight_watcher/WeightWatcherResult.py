from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pandas import DataFrame

from models import ModelIdentification


class WeightWatcherDetailsColumns(Enum):
    ACCURACY = "accuracy"
    ARCHITECTURE = "architecture"
    VARIANT = "variant"
    LAYER_ID = "layer_id"
    NAME = "name"
    D = "D"
    LAMBDA = "Lambda"
    M = "M"
    N = "N"
    ALPHA = "alpha"
    ALPHA_WEIGHTED = "alpha_weighted"
    BEST_FIT = "best_fit"
    ENTROPY = "entropy"
    HAS_ESD = "has_esd"
    LAMBDA_MAX = "lambda_max"
    LAYER_TYPE = "layer_type"
    LOG_ALPHA_NORM = "log_alpha_norm"
    LOG_NORM = "log_norm"
    LOG_SPECTRAL_NORM = "log_spectral_norm"
    MATRIX_RANK = "matrix_rank"
    NORM = "norm"
    NUM_EVALS = "num_evals"
    NUM_PL_SPIKES = "num_pl_spikes"
    RANK_LOSS = "rank_loss"
    RF = "rf"
    SIGMA = "sigma"
    SPECTRAL_NORM = "spectral_norm"
    STABLE_RANK = "stable_rank"
    SV_MAX = "sv_max"
    WARNING = "warning"
    WEAK_RANK_LOSS = "weak_rank_loss"
    XMAX = "xmax"
    XMIN = "xmin"


class WeightWatcherSummaryColumns(Enum):
    ACCURACY = "accuracy"
    ARCHITECTURE = "architecture"
    VARIANT = "variant"
    LOG_NORM = "log_norm"
    ALPHA = "alpha"
    ALPHA_WEIGHTED = "alpha_weighted"
    LOG_ALPHA_NORM = "log_alpha_norm"
    LOG_SPECTRAL_NORM = "log_spectral_norm"
    STABLE_RANK = "stable_rank"


@dataclass
class WeightWatcherResult:
    model_identification: ModelIdentification
    model_accuracy: float
    summary: DataFrame
    details: DataFrame
