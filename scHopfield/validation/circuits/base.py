from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
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
                 initial_state: Optional[np.ndarray] = None,
                 rtol: float = 1e-7, atol: float = 1e-9):
        """Integrate ``rhs`` over ``[0, t_end]`` from a single initial state.

        Parameters
        ----------
        t_end : float, optional (default: 200.0)
            End of the integration interval.
        n_samples : int, optional (default: 2000)
            Number of evenly spaced time points at which the solution is reported.
        initial_state : np.ndarray, optional
            State to start from, one value per entry of ``state_names``. It may be
            omitted only by circuits that define a canonical ``initial_conditions``
            mapping, such as a biophysical model with literature initial values.
            The synthetic circuits here define none: their initial state selects
            which attractor or phase the trajectory reaches, so no default would be
            safe. Draw one from ``sample_initial_conditions`` or pass an array.
        rtol, atol : float, optional
            Relative and absolute tolerances passed to ``solve_ivp``.

        Returns
        -------
        t : np.ndarray of shape (n_samples,)
            The sampled time points.
        x : np.ndarray of shape (n_samples, n_genes)
            The trajectory, one row per time point.

        Raises
        ------
        TypeError
            If ``initial_state`` is omitted and the circuit defines no
            ``initial_conditions``.
        """
        if initial_state is None:
            conditions = getattr(self, "initial_conditions", None)
            if conditions is None:
                name = type(self).__name__
                raise TypeError(
                    f"{name} defines no canonical initial_conditions, so simulate() "
                    "needs an explicit initial_state. Which attractor the trajectory "
                    "reaches is set by where it starts, so there is no safe default: "
                    f"draw one with {name}().sample_initial_conditions(1)[0], or pass "
                    "your own array of length "
                    f"{len(self.state_names)} ordered as {tuple(self.state_names)}."
                )
            initial_state = np.array([conditions[n] for n in self.state_names])
        sol = solve_ivp(
            lambda t, x: self.rhs(x),
            t_span=(0.0, t_end),
            y0=initial_state,
            t_eval=np.linspace(0.0, t_end, n_samples),
            method="LSODA", rtol=rtol, atol=atol,
        )
        return sol.t, sol.y.T
