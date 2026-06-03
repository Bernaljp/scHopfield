"""JAK2/STAT5 signaling in erythroid progenitor cells (Adlung et al. 2021).

Source: ported from ``paper/working/Adlung2021 _model_jakstat_pa.ode`` (XPPAUT),
itself an SBML auto-conversion of BIOMD0000001077 / Adlung et al. *Cell Reports*
36 (6), 2021, "Cell-to-cell variability in JAK2/STAT5 pathway components and
cytoplasmic volumes defines survival threshold in erythroid progenitor cells"
(PMID 34380040).

This circuit was chosen because:
1. It is a published, fitted model of a real signaling pathway.
2. STAT5 is a close paralog of STAT3, the lineage-balancing regulator our
   scHopfield perturbation analysis identified in hematopoiesis.
3. Erythroid progenitors are exactly the cells in our Paul15 hematopoiesis
   dataset, so this connects the validation directly to our applied work.

Topology
--------
External Epo (erythropoietin) drives ligand-bound EpoR/JAK2 phosphorylation,
which in turn phosphorylates STAT5, which translocates into the nucleus.
Nuclear pSTAT5 (npSTAT5) drives transcription of two negative-feedback
inhibitors: CIS and SOCS3. The mRNA delay chain CISnRNA1 -> CISnRNA2 -> CISRNA
models nuclear-export delay.

This is a transient response, not an oscillator. Stimulation by Epo produces
a single npSTAT5 pulse that decays via the negative feedback. The Hopfield
fit should capture the pulse shape.

State variables (14)
--------------------
* ``EpoRJAK2``        -- unphosphorylated EpoR/JAK2 complex
* ``EpoRpJAK2``       -- ligand-bound, phosphorylated EpoR/JAK2
* ``p1EpoRpJAK2``     -- singly phosphorylated EpoR/JAK2 form 1
* ``p2EpoRpJAK2``     -- singly phosphorylated EpoR/JAK2 form 2
* ``p12EpoRpJAK2``    -- doubly phosphorylated EpoR/JAK2
* ``SHP1Act``         -- active SHP1 phosphatase
* ``STAT5``           -- cytoplasmic unphosphorylated STAT5
* ``pSTAT5``          -- cytoplasmic phosphorylated STAT5
* ``npSTAT5``         -- nuclear phosphorylated STAT5  (the key output)
* ``CISnRNA1``        -- CIS pre-mRNA stage 1 (nucleus)
* ``CISnRNA2``        -- CIS pre-mRNA stage 2 (nucleus)
* ``CISRNA``          -- mature CIS mRNA (cytoplasm)
* ``CIS``             -- CIS protein
* ``SOCS3``           -- SOCS3 protein

Constants (non-dynamical species)
* ``SHP1`` = 26.7236  -- total SHP1 (not depleted by reactions in this model)

Default parameters from the BioModels entry; with Epo=20 (level) the system
produces a clear npSTAT5 pulse over ~120 time units.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict
import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_PARAMS: Dict[str, float] = {
    # Compartments
    "cyt":             0.4,
    "nuc":             0.275,
    # Stimulation / inhibitor levels (control parameters)
    "Epo":             20.0,    # external Epo level
    "ActD":            0.0,     # transcription inhibitor; 0 = no inhibition
    "SOCS3oe":         0.0,     # SOCS3 overexpression; 0 = baseline
    # Kinetic rates from Adlung 2021 best-fit
    "CISEqc":          0.767538787148837,
    "CISInh":          4.3834039529483e8,
    "CISRNADelay":     0.119845304696486,
    "CISRNAEqc":       1.0,
    "CISRNATurn":      0.119809412320528,
    "CISTurn":         0.0178232876161209,
    "DeaEpoRJAKActSHP1": 8.85910280224449e-4,
    "EpoRActJAK2":     0.326237934674659,
    "JAK2ActEpo":      0.0520769792397573,
    "SHP1ActEpoR":     1.0,
    "SHP1Dea":         0.00557392820000894,
    "SOCS3Eqc":        0.162493786913208,
    "SOCS3EqcOE":      828.06160444759,
    "SOCS3Inh":        0.010341350346111,
    "SOCS3Turn":       0.0806005449025786,
    "STAT5ActEpoR":    0.299136651563824,
    "STAT5ActJAK2":    0.0513253755320527,
    "STAT5Exp":        0.0112157105187786,
    "STAT5Imp":        0.0404763494591488,
    # Total SHP1 (non-dynamical)
    "SHP1":            26.7236153222782,
}


STATE_NAMES: Tuple[str, ...] = (
    "EpoRJAK2", "EpoRpJAK2", "p1EpoRpJAK2", "p2EpoRpJAK2", "p12EpoRpJAK2",
    "SHP1Act", "STAT5", "pSTAT5", "npSTAT5",
    "CISnRNA1", "CISnRNA2", "CISRNA", "CIS", "SOCS3",
)

DEFAULT_IC: Dict[str, float] = {
    "EpoRJAK2":     3.97504832099667,
    "EpoRpJAK2":    0.0,
    "p1EpoRpJAK2":  0.0,
    "p2EpoRpJAK2":  0.0,
    "p12EpoRpJAK2": 0.0,
    "SHP1Act":      0.0,
    "STAT5":        79.7242077843376,
    "pSTAT5":       0.0,
    "npSTAT5":      0.0,
    "CISnRNA1":     0.0,
    "CISnRNA2":     0.0,
    "CISRNA":       0.0,
    "CIS":          0.0,
    "SOCS3":        0.0,
}


@dataclass
class Adlung2021JakStat:
    """JAK2/STAT5 signaling in erythroid progenitor cells (BIOMD0000001077).

    Parameters
    ----------
    params : dict
        Kinetic parameters; defaults match the BioModels best-fit values.
    initial_conditions : dict
        Initial state.
    """

    params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    initial_conditions: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_IC))

    @property
    def n_genes(self) -> int:
        return len(STATE_NAMES)

    @property
    def gene_names(self) -> Tuple[str, ...]:
        return STATE_NAMES

    def _unpack(self, x: np.ndarray) -> Dict[str, float]:
        return dict(zip(STATE_NAMES, x))

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the ODE system.

        Directly translated from the 26 reactions (v1...v26) and the species
        ODEs in the .ode file. Compartment scaling 1/cyt or 1/nuc applied per
        the SBML rule.
        """
        p = self.params
        s = self._unpack(x)

        cyt = p["cyt"]; nuc = p["nuc"]
        SOCS3_inh_factor = s["SOCS3"] * p["SOCS3Inh"] / p["SOCS3Eqc"] + 1.0
        CIS_inh_factor = s["CIS"] * p["CISInh"] / p["CISEqc"] + 1.0

        # 26 reactions (named v1 ... v26 in the .ode file)
        v1  = cyt * (p["Epo"] * s["EpoRJAK2"] * p["JAK2ActEpo"]
                     / SOCS3_inh_factor)
        v2  = cyt * (p["DeaEpoRJAKActSHP1"] * s["EpoRpJAK2"] * s["SHP1Act"]
                     / p["SHP1ActEpoR"])
        v3  = cyt * (s["EpoRpJAK2"] * p["EpoRActJAK2"] / SOCS3_inh_factor)
        v4  = cyt * (3.0 * s["EpoRpJAK2"] * p["EpoRActJAK2"] / SOCS3_inh_factor)
        v5  = cyt * (3.0 * p["EpoRActJAK2"] * s["p1EpoRpJAK2"] / SOCS3_inh_factor)
        v6  = cyt * (p["EpoRActJAK2"] * s["p2EpoRpJAK2"] / SOCS3_inh_factor)
        v7  = cyt * (p["DeaEpoRJAKActSHP1"] * s["p1EpoRpJAK2"] * s["SHP1Act"]
                     / p["SHP1ActEpoR"])
        v8  = cyt * (p["DeaEpoRJAKActSHP1"] * s["p2EpoRpJAK2"] * s["SHP1Act"]
                     / p["SHP1ActEpoR"])
        v9  = cyt * (p["DeaEpoRJAKActSHP1"] * s["p12EpoRpJAK2"] * s["SHP1Act"]
                     / p["SHP1ActEpoR"])
        # v10, v11 missing in the .ode file (gap in BioModels numbering)
        SHP1_total = p["SHP1"]
        sum_pJAK2 = s["EpoRpJAK2"] + s["p1EpoRpJAK2"] + s["p2EpoRpJAK2"] + s["p12EpoRpJAK2"]
        v12 = cyt * (SHP1_total * p["SHP1ActEpoR"] * sum_pJAK2)
        v13 = cyt * (p["SHP1Dea"] * s["SHP1Act"])
        v14 = cyt * (s["STAT5"] * p["STAT5ActJAK2"] * sum_pJAK2 / SOCS3_inh_factor)
        v15 = cyt * (s["STAT5"] * p["STAT5ActEpoR"]
                     * (s["p12EpoRpJAK2"] + s["p1EpoRpJAK2"])**2
                     / (SOCS3_inh_factor * CIS_inh_factor))
        v16 = cyt * p["STAT5Imp"] * s["pSTAT5"]
        v17 = nuc * p["STAT5Exp"] * s["npSTAT5"]
        v18 = nuc * (-p["CISRNAEqc"]) * p["CISRNATurn"] * s["npSTAT5"] * (p["ActD"] - 1.0)
        v19 = nuc * p["CISRNADelay"] * s["CISnRNA1"]
        v20 = nuc * s["CISnRNA2"] * p["CISRNADelay"]
        v21 = cyt * p["CISRNATurn"] * s["CISRNA"]
        v22 = cyt * (s["CISRNA"] * p["CISEqc"] * p["CISTurn"])
        v23 = cyt * p["CISTurn"] * s["CIS"]
        v24 = cyt * (-p["SOCS3Eqc"]) * p["SOCS3Turn"] * s["npSTAT5"] * (p["ActD"] - 1.0)
        v25 = cyt * p["SOCS3Turn"] * s["SOCS3"]
        v26 = cyt * (p["SOCS3oe"] * p["SOCS3Eqc"] * p["SOCS3Turn"] * p["SOCS3EqcOE"])

        dx = np.zeros_like(x)
        i = {n: k for k, n in enumerate(STATE_NAMES)}

        dx[i["EpoRJAK2"]]     = (1.0/cyt) * (-v1 + v2 + v7 + v8 + v9)
        dx[i["EpoRpJAK2"]]    = (1.0/cyt) * ( v1 - v2 - v3 - v4)
        dx[i["p1EpoRpJAK2"]]  = (1.0/cyt) * ( v3 - v5 - v7)
        dx[i["p2EpoRpJAK2"]]  = (1.0/cyt) * ( v4 - v6 - v8)
        dx[i["p12EpoRpJAK2"]] = (1.0/cyt) * ( v5 + v6 - v9)
        dx[i["SHP1Act"]]      = (1.0/cyt) * ( v12 - v13)
        dx[i["STAT5"]]        = (1.0/cyt) * (-v14 - v15 + v17)
        dx[i["pSTAT5"]]       = (1.0/cyt) * ( v14 + v15 - v16)
        dx[i["npSTAT5"]]      = (1.0/nuc) * ( v16 - v17)
        dx[i["CISnRNA1"]]     = (1.0/nuc) * ( v18 - v19)
        dx[i["CISnRNA2"]]     = (1.0/nuc) * ( v19 - v20)
        dx[i["CISRNA"]]       = (1.0/cyt) * ( v20 - v21)
        dx[i["CIS"]]          = (1.0/cyt) * ( v22 - v23)
        dx[i["SOCS3"]]        = (1.0/cyt) * ( v24 - v25 + v26)
        return dx

    def simulate(self, t_end: float = 200.0, n_samples: int = 2000,
                 initial_state: np.ndarray = None,
                 rtol: float = 1e-7, atol: float = 1e-9):
        """Standard simulation: stimulate with Epo at t=0 from resting state."""
        if initial_state is None:
            initial_state = np.array([self.initial_conditions[n] for n in STATE_NAMES])
        sol = solve_ivp(
            lambda t, x: self.rhs(x),
            t_span=(0.0, t_end),
            y0=initial_state,
            t_eval=np.linspace(0.0, t_end, n_samples),
            method="LSODA", rtol=rtol, atol=atol,
        )
        return sol.t, sol.y.T

    def __repr__(self) -> str:
        return (f"Adlung2021JakStat(14 vars, Epo={self.params['Epo']:.1f}, "
                f"ActD={self.params['ActD']})")
