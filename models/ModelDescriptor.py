from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelDescriptor:
    id: str
    tags: Dict[str, Any]

    def __str__(self):
        return self.id

    def __hash__(self):
        return self.id.__hash__()
