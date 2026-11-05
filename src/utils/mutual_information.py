"""Calculates weighted mutual information and cross-chain coupling scores."""

import numpy as np

from src.utils import entropy

# Preserve the sequence-weight helper previously exposed by this module.
calculate_sequence_weights = entropy.calculate_sequence_weights


def select_mi_columns(
    encoded_msa: np.ndarray,
    weights: np.ndarray,
    min_coverage: float = 0.8,
) -> np.ndarray:
    """Selects variable alignment columns with sufficient weighted coverage.

    The first sequence is the reference. Columns must contain a standard amino
    acid in the reference and at least two distinct amino acids among sequences
    with positive weights.

    Args:
        encoded_msa: Nonempty integer array of shape (num_sequences,
            num_positions). Codes 0 through 19 represent standard amino acids;
            codes 20 and above represent gaps or unknown residues.
        weights: Nonnegative sequence weights of shape (num_sequences,) with a
            positive sum.
        min_coverage: Minimum fraction of total sequence weight contributed by
            standard amino acids in a column, including the threshold.

    Returns:
        A one-dimensional array of retained zero-based column indices in
        ascending order, referring to the original alignment.
    """
    valid = encoded_msa < 20
    coverage = (weights[:, None] * valid).sum(axis=0) / weights.sum()
    eligible = valid[0] & (coverage >= min_coverage)
    for j in np.flatnonzero(eligible):
        observed = encoded_msa[valid[:, j] & (weights > 0), j]
        eligible[j] = np.unique(observed).size >= 2
    return np.flatnonzero(eligible)


def calculate_mi_matrix(
    encoded_msa: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates weighted pairwise mutual information in bits and pair depths.

    Only sequences with standard amino acids at both positions contribute to a
    pair. Marginal frequencies come from the same joint frequency table, and no
    pseudocounts are added.

    Args:
        encoded_msa: Integer array of shape (num_sequences, num_positions).
            Codes 0 through 19 represent standard amino acids; codes 20 and
            above represent gaps or unknown residues.
        weights: Nonnegative sequence weights of shape (num_sequences,). Use
            weights calculated before filtering alignment columns.

    Returns:
        A tuple (raw_scores, pair_depths) of symmetric arrays, each with shape
        (num_positions, num_positions). Raw scores have a zero diagonal and NaN
        for pairs with zero effective depth. Pair depths sum the weights of
        jointly valid sequences; their diagonal contains each column's total
        valid weight. Neither input is modified.
    """
    msa = np.asarray(encoded_msa)
    weights = np.asarray(weights, dtype=np.float64)

    # Widen residue codes before encoding pairs to avoid uint8 overflow.
    msa = msa.astype(np.int64, copy=False)
    valid = msa < 20
    n_positions = msa.shape[1]
    raw_scores = np.full((n_positions, n_positions), np.nan)
    pair_depths = np.zeros_like(raw_scores)
    np.fill_diagonal(raw_scores, 0.0)
    np.fill_diagonal(pair_depths, (weights[:, None] * valid).sum(axis=0))

    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            jointly_valid = valid[:, i] & valid[:, j]
            pair_weights = weights[jointly_valid]
            depth = pair_weights.sum()
            pair_depths[i, j] = pair_depths[j, i] = depth
            if depth == 0:
                continue
            pair_codes = 20 * msa[jointly_valid, i] + msa[jointly_valid, j]
            joint_counts = np.bincount(
                pair_codes, weights=pair_weights, minlength=400
            )
            joint = joint_counts.reshape(20, 20) / depth
            independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
            positive = joint > 0
            mi = np.sum(
                joint[positive]
                * np.log2(joint[positive] / independent[positive])
            )
            raw_scores[i, j] = raw_scores[j, i] = max(float(mi), 0.0)
    return raw_scores, pair_depths


def apply_apc(raw_scores: np.ndarray) -> np.ndarray:
    """Applies average product correction using off-diagonal score means.

    Args:
        raw_scores: Symmetric pairwise score array of shape (num_positions,
            num_positions). A defined correction requires at least two
            positions, finite scores, and a nonzero off-diagonal overall mean.

    Returns:
        A new float64 array with the same shape and a zero diagonal. Each
        off-diagonal entry is reduced by the product of its row and column
        means divided by the overall mean. Negative corrected scores are
        retained. Undefined means can produce nonfinite off-diagonal values.
    """
    n_positions = raw_scores.shape[0]
    scores = np.array(raw_scores, dtype=np.float64, copy=True)

    np.fill_diagonal(scores, 0.0)

    row_means = scores.sum(axis=1) / (n_positions - 1)
    overall_mean = row_means.mean()

    scores -= np.outer(row_means, row_means) / overall_mean
    np.fill_diagonal(scores, 0.0)

    return scores


def calculate_cross_chain_mi(
    encoded_msa: np.ndarray,
    chain_length_a: int,
    min_pair_depth: float = 100.0,
) -> dict:
    """Scores each eligible chain A position by its three strongest B partners.

    Sequence weights are calculated from the full alignment. Average product
    correction is applied to the matrix of all retained A and B columns before
    selecting partners with sufficient effective pair depth.

    Args:
        encoded_msa: Nonempty integer alignment of shape (num_sequences,
            num_positions), with chain A followed by chain B and the reference
            sequence first. Codes 0 through 19 represent standard amino acids;
            codes 20 and above represent gaps or unknown residues.
        chain_length_a: Number of leading alignment columns belonging to
            chain A. All remaining columns belong to chain B.
        min_pair_depth: Minimum sum of sequence weights for rows valid at both
            positions of a pair, including the threshold.

    Returns:
        A dictionary mapping original zero-based chain A column indices to the
        mean of their three highest APC-corrected mutual information scores.
        Ineligible positions and positions with fewer than three qualifying
        chain B partners are omitted. Values can be NaN if APC is undefined.
    """
    weights = calculate_sequence_weights(encoded_msa)
    columns = select_mi_columns(encoded_msa, weights)

    a_indices = np.flatnonzero(columns < chain_length_a)
    b_indices = np.flatnonzero(columns >= chain_length_a)

    raw_mi, pair_depths = calculate_mi_matrix(encoded_msa[:, columns], weights)
    corrected = apply_apc(raw_mi)

    scores = {}
    for i in a_indices:
        original_a = columns[i]
        candidates = b_indices[pair_depths[i, b_indices] >= min_pair_depth]
        if candidates.size < 3:
            continue
        order = np.argsort(-corrected[i, candidates], kind="stable")[:3]
        partners = candidates[order]
        scores[original_a] = corrected[i, partners].mean()

    return scores
