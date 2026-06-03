"""Novak-Tyson 1997 fission yeast cell cycle (BioModels BIOMD0000000007).

Source: ported from ``paper/working/BIOMD0000000007.ode`` (XPPAUT) and
``paper/working/BIOMD0000000007-matlab.m`` (MATLAB), which are the SBML
auto-conversions of Novak & Tyson 1997 (PMID 9256450, PMID 10395816).

State (13 dynamical variables)
------------------------------
* ``UbE``    -- ubiquitin protease 1 (active fraction)
* ``UbE2``   -- ubiquitin protease 2 (active fraction)
* ``Wee1``   -- Wee1 kinase (active fraction)
* ``Cdc25``  -- Cdc25 phosphatase (active fraction)
* ``G2K``    -- Cdc13/Cdc2 complex (active MPF precursor)
* ``R``      -- free Rum1 (CDK inhibitor)
* ``G1K``    -- Cig2/Cdc2 complex (G1/S CDK)
* ``IE``     -- intermediary enzyme
* ``PG2``    -- Cdc13/P-Cdc2 (phosphorylated, inactive)
* ``G1R``    -- Cig2/Cdc2/Rum1 complex
* ``G2R``    -- Cdc13/Cdc2/Rum1 complex
* ``PG2R``   -- Cdc13/P-Cdc2/Rum1 complex
* ``Mass``   -- cell mass (grows exponentially, halved at division)

Assignment rules
----------------
Variables computed on-the-fly from the dynamical state and used inside the
reactions (these are NOT independent state variables):

* ``MPF = G2K + beta * PG2``           -- M-phase promoting factor
* ``SPF = Cig1 + alpha * G1K + MPF``   -- S-phase promoting factor
* ``Rum1Total = R + G1R + G2R + PG2R``
* ``IEB = 1 - IE`` (and similar for UbE, UbE2, Wee1, Cdc25)
* ``k2 = UbE * V2 + (1 - UbE) * V2'``   -- Cdc13 degradation rate
* ``k6 = UbE2 * V6 + (1 - UbE2) * V6'`` -- Cig2 degradation rate
* ``kwee = Vw' * (1 - Wee1) + Vw * Wee1``
* ``k25 = Cdc25 * V25 + (1 - Cdc25) * V25'``

Events
------
* ``Start``: fires once when ``SPF >= 0.1`` and halves ``kp``.
* ``Division``: fires when ``UbE <= 0.1`` and (doubles ``kp``, halves Mass).

In this implementation we skip the Start event by initializing ``kp =
3.25 / 2 = 1.625``, matching the post-Start parameter; the Division event
is handled by the simulator loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .base import BaseCircuit
from typing import Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp


# Default parameter set straight from BIOMD0000000007. Names match the
# .ode file. After the one-time Start event, kp is halved from 3.25 to 1.625;
# we initialize at the halved value to avoid the event.
DEFAULT_PARAMS: Dict[str, float] = {
    "mu":       0.00495,
    "k1":       0.015,
    "k2prime":  0.05,
    "k3":       0.09375,
    "k4":       0.1875,
    "k5":       0.00175,
    "k6prime":  0.0,
    "k7":       100.0,
    "k7r":      0.1,
    "k8":       10.0,
    "k8r":      0.1,
    "kc":       1.0,
    "kcr":      0.25,
    "ki":       0.4,
    "kir":      0.1,
    "kp":       1.625,    # half of 3.25 (post-Start value)
    "ku":       0.2,
    "kur":      0.1,
    "ku2":      1.0,
    "kur2":     0.3,
    "kw":       1.0,
    "kwr":      0.25,
    "V2":       0.25,
    "V2prime":  0.0075,
    "V6":       7.5,
    "V6prime":  0.0375,
    "V25":      0.5,
    "V25prime": 0.025,
    "Vw":       0.35,
    "Vwprime":  0.035,
    "Kmc":      0.1,
    "Kmcr":     0.1,
    "Kmi":      0.01,
    "Kmir":     0.01,
    "Kmp":      0.001,
    "Kmu":      0.01,
    "Kmur":     0.01,
    "Kmu2":     0.05,
    "Kmur2":    0.05,
    "Kmw":      0.1,
    "Kmwr":     0.1,
    "alpha":    0.25,
    "beta":     0.05,
    "Cig1":     0.0,
}

# Initial conditions, same order as STATE_NAMES.
DEFAULT_IC: Dict[str, float] = {
    "UbE":      0.11,
    "UbE2":     0.0,
    "Wee1":     0.0,
    "Cdc25":    0.0,
    "G2K":      0.0,
    "R":        0.4,
    "G1K":      0.0,
    "IE":       0.0,
    "PG2":      0.0,
    "G1R":      0.0,
    "G2R":      0.0,
    "PG2R":     0.0,
    "Mass":     0.49,
}

STATE_NAMES: Tuple[str, ...] = tuple(DEFAULT_IC.keys())


@dataclass
class Novak1997CellCycle(BaseCircuit):
    """Fission yeast cell cycle from BIOMD0000000007 (Novak-Tyson 1997).

    Parameters are stored as a mutable dict so the Division event can mutate
    ``kp`` and the simulator can mutate ``Mass`` directly on the state.

    Use :meth:`simulate` to integrate with event handling, then pull the
    resulting (state, dx/dt) trajectories.
    """

    params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    initial_conditions: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_IC))



    # ----------------------------------------------------------------- rhs --


    @property
    def state_names(self) -> Tuple[str, ...]:
        return STATE_NAMES

    def rhs(self, x: np.ndarray, params: Dict[str, float] = None) -> np.ndarray:
        """dx/dt for the 13 dynamical variables. Mirrors xdot() in the .m file."""
        p = params if params is not None else self.params
        s = self._unpack(x)

        # Assignment rules
        MPF = s["G2K"] + p["beta"] * s["PG2"]
        SPF = p["Cig1"] + p["alpha"] * s["G1K"] + MPF
        IEB = 1.0 - s["IE"]
        UbEB = 1.0 - s["UbE"]
        UbE2B = 1.0 - s["UbE2"]
        Wee1B = 1.0 - s["Wee1"]
        Cdc25B = 1.0 - s["Cdc25"]
        k2 = s["UbE"] * p["V2"] + (1.0 - s["UbE"]) * p["V2prime"]
        k6 = s["UbE2"] * p["V6"] + (1.0 - s["UbE2"]) * p["V6prime"]
        kwee = p["Vwprime"] * (1.0 - s["Wee1"]) + p["Vw"] * s["Wee1"]
        k25 = s["Cdc25"] * p["V25"] + (1.0 - s["Cdc25"]) * p["V25prime"]

        # Reactions
        r_G2K_Creation       = p["k1"]
        r_G1K_Creation       = p["k5"]
        r_Cdc2Phos           = s["G2K"] * kwee - k25 * s["PG2"]
        r_G2R_Creation       = s["G2K"] * p["k7"] * s["R"] - s["G2R"] * p["k7r"]
        r_PG2R_Creation      = p["k7"] * s["PG2"] * s["R"] - p["k7r"] * s["PG2R"]
        r_Rum1DegInG2R       = s["G2R"] * p["k4"]
        r_Rum1Deg            = p["k4"] * s["R"]
        r_Rum1DegInPG2R      = p["k4"] * s["PG2R"]
        r_RumDegInG1R        = s["G1R"] * p["k4"]
        r_G2K_dissoc         = s["G2K"] * k2
        r_PG2_dissoc         = k2 * s["PG2"]
        r_G1K_Dissociation   = s["G1K"] * k6
        r_PG2R_Dissociation  = p["k2prime"] * s["PG2R"]
        r_G2R_Dissociation   = s["G2R"] * p["k2prime"]
        r_G1R_Dissociation   = s["G1R"] * p["k6prime"]
        r_G1R_Binding        = s["G1K"] * p["k8"] * s["R"] - s["G1R"] * p["k8r"]
        r_G2R_Dissoc_UbE     = s["G2R"] * k2
        r_PG2R_Dissoc_UbE    = k2 * s["PG2R"]
        r_Rum1_Production    = p["k3"]
        r_Rum1_Deg_SPF       = p["kp"] * s["Mass"] * s["R"] * SPF / (p["Kmp"] + s["R"])
        r_IE_Reaction        = (IEB * p["ki"] * MPF / (IEB + p["Kmi"])
                                - s["IE"] * p["kir"] / (s["IE"] + p["Kmir"]))
        r_UbE_Reaction       = (s["IE"] * p["ku"] * UbEB / (p["Kmu"] + UbEB)
                                - p["kur"] * s["UbE"] / (p["Kmur"] + s["UbE"]))
        r_UbE2_Reaction      = (p["ku2"] * MPF * UbE2B / (p["Kmu2"] + UbE2B)
                                - p["kur2"] * s["UbE2"] / (p["Kmur2"] + s["UbE2"]))
        r_Wee1_Reaction      = (p["kwr"] * Wee1B / (p["Kmwr"] + Wee1B)
                                - p["kw"] * MPF * s["Wee1"] / (p["Kmw"] + s["Wee1"]))
        r_Cdc25_Reaction     = (Cdc25B * p["kc"] * MPF / (Cdc25B + p["Kmc"])
                                - s["Cdc25"] * p["kcr"] / (s["Cdc25"] + p["Kmcr"]))

        dx = np.zeros_like(x)
        dx[STATE_NAMES.index("UbE")]   = r_UbE_Reaction
        dx[STATE_NAMES.index("UbE2")]  = r_UbE2_Reaction
        dx[STATE_NAMES.index("Wee1")]  = r_Wee1_Reaction
        dx[STATE_NAMES.index("Cdc25")] = r_Cdc25_Reaction
        dx[STATE_NAMES.index("G2K")]   = (r_G2K_Creation - r_Cdc2Phos
                                          - r_G2R_Creation + r_Rum1DegInG2R
                                          - r_G2K_dissoc)
        dx[STATE_NAMES.index("R")]     = (- r_G2R_Creation - r_PG2R_Creation
                                          - r_Rum1Deg + r_PG2R_Dissociation
                                          + r_G2R_Dissociation + r_G1R_Dissociation
                                          - r_G1R_Binding + r_G2R_Dissoc_UbE
                                          + r_PG2R_Dissoc_UbE + r_Rum1_Production
                                          - r_Rum1_Deg_SPF)
        dx[STATE_NAMES.index("G1K")]   = (r_G1K_Creation + r_RumDegInG1R
                                          - r_G1K_Dissociation - r_G1R_Binding)
        dx[STATE_NAMES.index("IE")]    = r_IE_Reaction
        dx[STATE_NAMES.index("PG2")]   = (r_Cdc2Phos - r_PG2R_Creation
                                          + r_Rum1DegInPG2R - r_PG2_dissoc)
        dx[STATE_NAMES.index("G1R")]   = (- r_RumDegInG1R - r_G1R_Dissociation
                                          + r_G1R_Binding)
        dx[STATE_NAMES.index("G2R")]   = (r_G2R_Creation - r_Rum1DegInG2R
                                          - r_G2R_Dissociation - r_G2R_Dissoc_UbE)
        dx[STATE_NAMES.index("PG2R")]  = (r_PG2R_Creation - r_Rum1DegInPG2R
                                          - r_PG2R_Dissociation - r_PG2R_Dissoc_UbE)
        dx[STATE_NAMES.index("Mass")]  = s["Mass"] * p["mu"]
        return dx

    # -------------------------------------------------------------- simulate --

    def simulate(self, t_end: float = 400.0, n_samples: int = 4000,
                  initial_state: np.ndarray = None,
                  rtol: float = 1e-7, atol: float = 1e-9,
                  max_divisions: int = 100,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate with Division event handling. Returns ``(t, X)`` where
        ``X`` has shape ``(n_samples, n_genes)``.

        The integrator stops whenever ``UbE`` crosses ``0.1`` from above, applies
        the Division reset (``Mass /= 2``, the kp parameter would double in the
        original model but we keep it fixed since the Division event simply
        marks the end of M-phase in this BioModels translation), and restarts.
        """
        if initial_state is None:
            initial_state = np.array([self.initial_conditions[n] for n in STATE_NAMES])

        t_dense = np.linspace(0.0, t_end, n_samples)
        X_dense = np.full((n_samples, self.n_genes), np.nan)

        def division_event(t, x):
            return x[STATE_NAMES.index("UbE")] - 0.1
        division_event.terminal = True
        division_event.direction = -1   # crossing from above to below

        t_cursor = 0.0
        state = initial_state.copy()
        n_div = 0
        while t_cursor < t_end and n_div < max_divisions:
            sol = solve_ivp(
                lambda t, x: self.rhs(x),
                t_span=(t_cursor, t_end),
                y0=state,
                t_eval=t_dense[(t_dense >= t_cursor) & (t_dense <= t_end)],
                events=division_event,
                rtol=rtol, atol=atol,
                method="LSODA",
            )
            # Fill the dense grid for the segment that was integrated.
            seg_t = sol.t
            seg_X = sol.y.T
            mask = (t_dense >= seg_t[0]) & (t_dense <= seg_t[-1])
            if mask.sum() > 0:
                X_dense[mask] = seg_X[:mask.sum()]
            if len(sol.t_events[0]) > 0:
                # Division event fired.
                t_cursor = sol.t_events[0][0]
                state = sol.y_events[0][0].copy()
                state[STATE_NAMES.index("Mass")] /= 2.0
                # Push UbE just below the threshold to avoid immediate retrigger.
                state[STATE_NAMES.index("UbE")] = 0.099
                n_div += 1
            else:
                break
        return t_dense, X_dense

    # ------------------------------------------------------- helpers / aux --

    def assignment_rules(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute MPF, SPF, Rum1Total, Cdc13Total along the trajectory."""
        p = self.params
        G2K = X[:, STATE_NAMES.index("G2K")]
        PG2 = X[:, STATE_NAMES.index("PG2")]
        G1K = X[:, STATE_NAMES.index("G1K")]
        R = X[:, STATE_NAMES.index("R")]
        G1R = X[:, STATE_NAMES.index("G1R")]
        G2R = X[:, STATE_NAMES.index("G2R")]
        PG2R = X[:, STATE_NAMES.index("PG2R")]
        MPF = G2K + p["beta"] * PG2
        SPF = p["Cig1"] + p["alpha"] * G1K + MPF
        return {
            "MPF": MPF,
            "SPF": SPF,
            "Rum1Total": R + G1R + G2R + PG2R,
            "Cdc13Total": G2K + PG2 + G2R + PG2R,
            "Cig2Total": G1K + G1R,
        }

    def __repr__(self) -> str:
        return (f"Novak1997CellCycle(13 vars, kp={self.params['kp']:.3f}, "
                f"mu={self.params['mu']:.5f})")
