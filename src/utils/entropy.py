"""Calculates sequence weights and site entropy for encoded alignments.

Residues use integer codes 0 through 19 for the standard amino acids. Codes 20
and above represent gaps or unknown residues and are excluded from site counts.
"""

import numba
import numpy as np


@numba.njit
def pair_identity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Calculates identity over positions with at least one valid residue.

    Args:
        seq_a: One-dimensional encoded sequence with nonnegative residue codes.
        seq_b: One-dimensional encoded sequence with the same length as seq_a.

    Returns:
        The fraction of compared positions containing matching standard amino
        acids. A gap or unknown residue in only one sequence counts as a
        mismatch.

    Raises:
        ZeroDivisionError: If neither sequence contains a standard amino acid.
    """
    valid_a = seq_a < 20
    valid_b = seq_b < 20

    compared = valid_a | valid_b
    matched = valid_a & valid_b & (seq_a == seq_b)

    n_compared = np.count_nonzero(compared)
    n_matched = np.count_nonzero(matched)

    return n_matched / n_compared


@numba.njit
def calculate_sequence_weights(
    encoded_msa: np.ndarray,
    identity_threshold: float = 0.8,
) -> np.ndarray:
    """Calculates inverse neighbor counts to downweight similar sequences.

    Args:
        encoded_msa: Integer alignment of shape (n_sequences, n_positions), with
            nonnegative residue codes. Codes below 20 are standard amino acids.
        identity_threshold: Minimum pairwise identity for counting two sequences
            as neighbors, including equality at the threshold.

    Returns:
        A float array of shape (n_sequences,) containing one divided by each
        sequence's neighbor count. Each sequence counts as its own neighbor.

    Raises:
        ZeroDivisionError: If a compared pair has no standard amino acids.
    """
    n_sequences = encoded_msa.shape[0]
    neighbor_counts = np.ones(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            identity = pair_identity(encoded_msa[i], encoded_msa[j])
            if identity >= identity_threshold:
                neighbor_counts[i] += 1
                neighbor_counts[j] += 1

    return 1.0 / neighbor_counts


def calculate_site_entropy(
    encoded_msa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates normalized site entropy and weighted amino acid counts.

    Sequence weights use an identity threshold of 0.8. Gaps and unknown residues
    do not contribute to counts or to the normalization of site frequencies.

    Args:
        encoded_msa: Integer alignment of shape (n_sequences, n_positions), with
            nonnegative residue codes. Codes below 20 are standard amino acids.

    Returns:
        A tuple containing entropies of shape (n_positions,) and weighted counts
        of shape (n_positions, 20). Entropy is divided by log(20), giving values
        from zero to one. Sites with no valid residues have NaN entropy and zero
        counts.

    Raises:
        ZeroDivisionError: If sequence weighting compares a pair with no
            standard amino acids.
    """
    weights = calculate_sequence_weights(encoded_msa)
    n_sequences, n_positions = encoded_msa.shape

    counts = np.zeros((n_positions, 20), dtype=np.float64)
    for i in range(n_sequences):
        for j in range(n_positions):
            amino_acid = encoded_msa[i, j]
            if amino_acid < 20:
                counts[j, amino_acid] += weights[i]

    frequencies = np.full((n_positions, 20), np.nan, dtype=np.float64)
    for j in range(n_positions):
        total_weight = counts[j].sum()
        if total_weight > 0:
            frequencies[j] = counts[j] / total_weight

    entropies = np.full(n_positions, np.nan, dtype=np.float64)

    for j in range(n_positions):
        site_frequencies = frequencies[j]
        if np.isnan(site_frequencies).any():
            continue
        # Zero-probability terms contribute zero without evaluating log(0).
        positive_frequencies = site_frequencies[site_frequencies > 0]
        entropy = -np.sum(positive_frequencies * np.log(positive_frequencies))
        entropies[j] = entropy / np.log(20)

    return entropies, counts
