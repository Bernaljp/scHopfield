"""Canonical synthetic circuits used to validate scHopfield representation
and inference. Each module exposes a Circuit class with a uniform API."""

from .toggle import ToggleCircuit
from .oscillator import OscillatorCircuit

__all__ = [
    "ToggleCircuit",
    "OscillatorCircuit",
]
