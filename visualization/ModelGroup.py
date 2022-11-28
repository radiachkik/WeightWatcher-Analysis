from dataclasses import dataclass
from typing import Callable

from ww import WWResult


@dataclass
class ModelGroup:
    name: str
    selector: Callable[[WWResult], bool]
