from __future__ import annotations

from enum import Enum, auto


class ParadigmStrategy(Enum):
    DIGITAL = auto()
    ANALOG = auto()
    SDAQC = auto()
    BDAQC = auto()
    RYDBERG = auto()
