import numpy as np
from src.utils.entropy import calculate_sequence_weights

def select_mi_columns(
    encoded_msa: np.ndarray,
    weights: np.ndarray,
    min_coverage: float = 0.8,
) -> np.ndarray:
    """"""
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
    """Return symmetric raw MI (bits) and effective pair-depth matrices.

    Supply retained A+B columns and weights computed BEFORE column filtering.
    Only jointly valid rows contribute; marginals come from the joint table.
    No pseudocounts are added. MI has a zero diagonal; depth has each column's
    valid weight on its diagonal. Unobserved pairs have NaN MI and zero depth.
    """
    msa = np.asarray(encoded_msa)
    weights = np.asarray(weights, dtype=np.float64)

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
            joint = np.bincount(
                pair_codes, weights=pair_weights, minlength=400
            ).reshape(20, 20) / depth
            independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
            positive = joint > 0
            mi = np.sum(joint[positive] * np.log2(
                joint[positive] / independent[positive]
            ))
            raw_scores[i, j] = raw_scores[j, i] = max(float(mi), 0.0)
    return raw_scores, pair_depths


def apply_apc(raw_scores: np.ndarray) -> np.ndarray:
    """"""
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
    """"""
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
