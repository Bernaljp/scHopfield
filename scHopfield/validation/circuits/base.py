from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp

class BaseCircuit(ABC):
    @property
    @abstractmethod
    def state_names(self) -> Tuple[str, ...]:
        pass
        
    @property
    def n_genes(self) -> int:
        return len(self.state_names)

    @property
    def gene_names(self) -> Tuple[str, ...]:
        return self.state_names

    def _unpack(self, x: np.ndarray) -> Dict[str, float]:
        return dict(zip(self.state_names, x))

    @abstractmethod
    def rhs(self, x: np.ndarray) -> np.ndarray:
        pass

    def simulate(self, t_end: float = 200.0, n_samples: int = 2000,
                 initial_state: np.ndarray = None,
                 rtol: float = 1e-7, atol: float = 1e-9):
        if initial_state is None:
            initial_state = np.array([self.initial_conditions[n] for n in self.state_names])
        sol = solve_ivp(
            lambda t, x: self.rhs(x),
            t_span=(0.0, t_end),
            y0=initial_state,
            t_eval=np.linspace(0.0, t_end, n_samples),
            method="LSODA", rtol=rtol, atol=atol,
        )
        return sol.t, sol.y.T
