"""
Energy calculation functionality for scHopfield package.
Contains classes and methods for computing various types of energy in the landscape.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any

try:
    from ..core.base_models import BaseEnergyCalculator
    from ..utils.utilities import to_numpy, int_sig_act_inv
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base_models import BaseEnergyCalculator
    from utils.utilities import to_numpy, int_sig_act_inv


class EnergyCalculator(BaseEnergyCalculator):
    """
    Calculator for various energy components in the Hopfield-like system.

    This class handles the computation of interaction, degradation, and bias energies
    for different clusters or cell types.
    """

    def __init__(self, analyzer):
        """
        Initialize the energy calculator with a reference to the main analyzer.

        Args:
            analyzer: The main LandscapeAnalyzer instance.
        """
        self.analyzer = analyzer

    def get_energies(self, x: Optional[np.ndarray] = None) -> Optional[Tuple[Dict, Dict, Dict, Dict]]:
        """
        Calculate and store the energies for each cluster or for a specific input x.

        Args:
            x: If provided, calculates energies for this specific input instead of the entire dataset.

        Returns:
            If x is provided, returns a tuple of dictionaries (E, E_interaction, E_degradation, E_bias),
            otherwise, updates the analyzer attributes with the calculated energies.
        """
        # Initialize dictionaries to store energies
        energies = {}
        interaction_energies = {}
        degradation_energies = {}
        bias_energies = {}

        # Iterate over each cluster to calculate energies
        for cluster in self.analyzer.W.keys():
            # Calculate each component of the energy for the current cluster
            interaction_energy = self.interaction_energy(cluster, x=x)
            degradation_energy = self.degradation_energy(cluster, x=x)
            bias_energy = self.bias_energy(cluster, x=x)

            # Total energy is the sum of all components
            total_energy = interaction_energy + degradation_energy + bias_energy

            # Store the calculated energies
            interaction_energies[cluster] = interaction_energy
            degradation_energies[cluster] = degradation_energy
            bias_energies[cluster] = bias_energy
            energies[cluster] = total_energy

        # If x is None, update analyzer attributes with the calculated energies
        if x is None:
            self.analyzer.E = energies
            self.analyzer.E_interaction = interaction_energies
            self.analyzer.E_degradation = degradation_energies
            self.analyzer.E_bias = bias_energies
        else:
            # If x is provided, return the calculated energies as a tuple of dictionaries
            return energies, interaction_energies, degradation_energies, bias_energies

    def write_energies(self) -> None:
        """
        Writes the calculated energies into the AnnData object as observations.
        """
        # Initialize energy columns in the AnnData observations with zeros
        self.analyzer.adata.obs['Total_energy'] = np.zeros(self.analyzer.adata.n_obs, dtype=float)
        self.analyzer.adata.obs['Interaction_energy'] = np.zeros(self.analyzer.adata.n_obs, dtype=float)
        self.analyzer.adata.obs['Degradation_energy'] = np.zeros(self.analyzer.adata.n_obs, dtype=float)
        self.analyzer.adata.obs['Bias_energy'] = np.zeros(self.analyzer.adata.n_obs, dtype=float)

        # Iterate over each cluster (excluding 'all') and update the energy values for cells in that cluster
        for cluster in [k for k in self.analyzer.E if k != 'all']:
            # Identify the cells belonging to the current cluster
            cluster_indices = self.analyzer.adata.obs[self.analyzer.cluster_key] == cluster

            # Update energy values for cells in the current cluster
            self.analyzer.adata.obs.loc[cluster_indices, 'Total_energy'] = self.analyzer.E[cluster]
            self.analyzer.adata.obs.loc[cluster_indices, 'Interaction_energy'] = self.analyzer.E_interaction[cluster]
            self.analyzer.adata.obs.loc[cluster_indices, 'Degradation_energy'] = self.analyzer.E_degradation[cluster]
            self.analyzer.adata.obs.loc[cluster_indices, 'Bias_energy'] = self.analyzer.E_bias[cluster]

    def interaction_energy(self, cl: str, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the interaction energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            x: Optional specific input to calculate energy for.

        Returns:
            Calculated interaction energy.
        """
        # Determine the indices for the cluster or use all indices
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        W = self.analyzer.W[cl]

        # Calculate the interaction energy
        interaction_energy = -0.5 * np.sum((sig @ W.T) * sig, axis=1)
        return interaction_energy

    def degradation_energy(self, cl: str, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the degradation energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            x: Optional specific input to calculate energy for.

        Returns:
            Calculated degradation energy.
        """
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        g = self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values if not self.analyzer.refit_gamma else self.analyzer.gamma[cl]

        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        integral = int_sig_act_inv(sig, self.analyzer.threshold, self.analyzer.exponent)
        degradation_energy = np.sum(g[None, :] * integral, axis=1)

        return degradation_energy

    def bias_energy(self, cl: str, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the bias energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            x: Optional specific input to calculate energy for.

        Returns:
            Calculated bias energy.
        """
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        I = self.analyzer.I[cl]

        # Calculate the bias energy
        bias_energy = -np.sum(I[None, :] * sig, axis=1)
        return bias_energy

    def degradation_energy_decomposed(self, cl: str, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the decomposition of degradation energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            x: Optional specific input to calculate energy for.

        Returns:
            Decomposed degradation energy for each gene.
        """
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        g = self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values if not self.analyzer.refit_gamma else self.analyzer.gamma[cl]

        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        integral = int_sig_act_inv(sig, self.analyzer.threshold, self.analyzer.exponent)
        return g[None, :] * integral

    def bias_energy_decomposed(self, cl: str, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the decomposition of bias energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            x: Optional specific input to calculate energy for.

        Returns:
            Decomposed bias energy for each gene.
        """
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        I = self.analyzer.I[cl]
        return -I[None, :] * sig

    def interaction_energy_decomposed(self, cl: str, side: str = 'in', x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the decomposition of interaction energy for a given cluster or all cells.

        Args:
            cl: The cluster identifier or 'all' for all cells.
            side: Specifies the side of the interaction energy to decompose ('in' or 'out').
            x: Optional specific input to calculate energy for.

        Returns:
            Decomposed interaction energy for each gene.
        """
        idx = self.analyzer.adata.obs[self.analyzer.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.analyzer.get_sigmoid(x) if x is not None else self.analyzer.get_matrix('sigmoid', genes=self.analyzer.genes)[idx]
        W = self.analyzer.W[cl]
        return -0.5*(sig @ W.T) * sig if side == 'out' else -0.5*(sig @ W) * sig