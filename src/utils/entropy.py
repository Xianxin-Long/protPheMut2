import numpy as np
from numba import njit

@njit
def pair_identity(
    seq_a: np.ndarray, 
    seq_b: np.ndarray
) -> float:
    """"""
    valid_a = seq_a < 20
    valid_b = seq_b < 20

    compared = valid_a | valid_b
    matched = valid_a & valid_b & (seq_a == seq_b)

    n_compared = np.count_nonzero(compared)
    n_matched = np.count_nonzero(matched)

    return n_matched / n_compared


@njit
def calculate_sequence_weights(
    encoded_msa: np.ndarray,
    identity_threshold: float = 0.8,
) -> np.ndarray:
    """"""
    n_sequences = encoded_msa.shape[0]
    neighbor_counts = np.ones(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            identity= pair_identity(encoded_msa[i], encoded_msa[j])
            if identity >= identity_threshold:
                neighbor_counts[i] += 1
                neighbor_counts[j] += 1

    return 1.0 / neighbor_counts


def calculate_site_entropy(encoded_msa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """"""
    # calculate weighted frequencies
    weights = calculate_sequence_weights(encoded_msa)
    n_sequences, n_positions = encoded_msa.shape

    counts = np.zeros((n_positions, 20), dtype=np.float64,)
    for i in range(n_sequences):
        for j in range(n_positions):
            aa = encoded_msa[i, j]
            if aa < 20:
                counts[j, aa] += weights[i]

    frequencies = np.full((n_positions, 20), np.nan, dtype=np.float64)
    for j in range(n_positions):
        total_weight = counts[j].sum()
        if total_weight > 0:
            frequencies[j] = counts[j] / total_weight

    # calculate entropy
    n_positions = frequencies.shape[0]
    entropies = np.full(n_positions, np.nan, dtype=np.float64)

    for j in range(n_positions):
        p = frequencies[j]
        if np.isnan(p).any():
            continue
        p_positive = p[p > 0]
        entropy = -np.sum(p_positive * np.log(p_positive))
        entropies[j] = entropy / np.log(20)

    return entropies, counts
