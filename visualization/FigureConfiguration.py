from dataclasses import dataclass
from typing import List

from visualization import PlotConfiguration


@dataclass
class FigureConfiguration:
    figure_name: str
    num_rows: int
    num_cols: int
    plot_configs: List[PlotConfiguration]
    width: int
