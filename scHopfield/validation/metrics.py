"""Metrics for comparing an inferred interaction matrix against ground truth.

Each function takes ``W_inferred`` and ``W_true`` (both ``(n_genes, n_genes)``)
and returns a scalar. ``summarize_recovery`` aggregates them.
"""
from __future__ import annotations
from typing import Dict
import numpy as np


def edge_sign_accuracy(W_inferred: np.ndarray, W_true: np.ndarray,
                       threshold: float = 1e-6) -> float:
    """Fraction of nonzero ground-truth edges whose sign is correctly inferred.

    An "edge" is any position where ``|W_true| > threshold``. For each such
    position we check whether ``sign(W_inferred) == sign(W_true)``. Zero entries
    in W_true are ignored.
    """
    mask = np.abs(W_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    correct = (np.sign(W_inferred[mask]) == np.sign(W_true[mask])).sum()
    return float(correct) / float(mask.sum())


def edge_signed_correlation(W_inferred: np.ndarray, W_true: np.ndarray) -> float:
    """Pearson correlation between flattened entries of W_inferred and W_true."""
    a = W_inferred.flatten()
    b = W_true.flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spectral_overlap(W_inferred: np.ndarray, W_true: np.ndarray) -> float:
    """Overlap of eigenvalue spectra, measured as 1 - normalized Hausdorff distance.

    Sorts both spectra by magnitude, pairs them up, and reports the average
    relative pairwise distance. Higher = better overlap. 1.0 = identical.
    """
    eig_a = np.sort(np.linalg.eigvals(W_inferred))[::-1]
    eig_b = np.sort(np.linalg.eigvals(W_true))[::-1]
    # Compare absolute and complex separately
    abs_a = np.sort(np.abs(eig_a))[::-1]
    abs_b = np.sort(np.abs(eig_b))[::-1]
    denom = np.maximum(abs_b, 1e-6)
    rel_diff = np.abs(abs_a - abs_b) / denom
    return float(1.0 - rel_diff.mean())


def symmetry_index(W: np.ndarray) -> float:
    """||W - W^T||_F / (||W + W^T||_F + epsilon).

    Pure symmetric matrices give 0; pure antisymmetric matrices give large
    values. Useful for showing that the repressilator's W is recovered with
    a dominant antisymmetric component.
    """
    sym = W + W.T
    anti = W - W.T
    return float(np.linalg.norm(anti, ord="fro") / (np.linalg.norm(sym, ord="fro") + 1e-12))


def frobenius_distance(W_inferred: np.ndarray, W_true: np.ndarray,
                       normalize: bool = True) -> float:
    """Frobenius distance between inferred and true W. Normalized by ||W_true||."""
    d = np.linalg.norm(W_inferred - W_true, ord="fro")
    if normalize:
        d = d / (np.linalg.norm(W_true, ord="fro") + 1e-12)
    return float(d)


def summarize_recovery(W_inferred: np.ndarray, W_true: np.ndarray) -> Dict[str, float]:
    """Aggregate the standard recovery metrics into one dict for table rendering."""
    return {
        "edge_sign_accuracy": edge_sign_accuracy(W_inferred, W_true),
        "edge_correlation": edge_signed_correlation(W_inferred, W_true),
        "spectral_overlap": spectral_overlap(W_inferred, W_true),
        "frobenius_distance": frobenius_distance(W_inferred, W_true),
        "symmetry_inferred": symmetry_index(W_inferred),
        "symmetry_true": symmetry_index(W_true),
    }
