from dataclasses import dataclass
from typing import Callable, Optional
from ww import WWResult


@dataclass
class PlotConfiguration:
    plot_name: str
    legend_group: str
    result_selector: Callable[[WWResult], bool]
    x_data_selector: Callable[[WWResult], float]
    y_data_selector: Callable[[WWResult], float]
    text_data_selector: Optional[Callable[[WWResult], str]]
    markers: bool
    text: bool
    lines: bool
    marker_size: Optional[int]
    x_axis_title: str
    y_axis_title: str
