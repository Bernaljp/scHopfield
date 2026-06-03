"""Canonical synthetic circuits used to validate scHopfield representation
and inference. Each module exposes a Circuit class with a uniform API."""

from .toggle import ToggleCircuit
from .oscillator import OscillatorCircuit, DissertationOscillatorCircuit
from .cell_cycle import Novak1997CellCycle, STATE_NAMES as CELL_CYCLE_STATE_NAMES
from .jakstat import Adlung2021JakStat, STATE_NAMES as JAKSTAT_STATE_NAMES

__all__ = [
    "ToggleCircuit",
    "OscillatorCircuit",
    "DissertationOscillatorCircuit",
    "Novak1997CellCycle",
    "CELL_CYCLE_STATE_NAMES",
    "Adlung2021JakStat",
    "JAKSTAT_STATE_NAMES",
]
